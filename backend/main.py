"""
FastAPI Backend Server for Cancer Variant Analysis
Production-ready with comprehensive error handling and logging
VCF / VCF.GZ compatible with graceful handling
"""

import json
import asyncio
import logging
import tempfile
import os
import traceback
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple
import uuid

import pandas as pd
import pysam
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from variant_agents import (
    filter_variants,
    analyze_variants,
    wilms_literature_reasoning,
    normalize_annovar_columns,
)
from panels.wilms_panel import load_wilms_panel
from burden_engine import annotate_wilms_burden, wilms_gene_burden

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
load_dotenv()

VALID_EXTENSIONS = (".vcf", ".vcf.gz", ".txt", ".tsv", ".csv")
REQUIRED_ANNOVAR_COLS = {
    "Gene.refGene", "Gene.refGeneWithVer",
    "ExonicFunc.refGene", "ExonicFunc.refGeneWithVer",
    "AAChange.refGene", "AAChange.refGeneWithVer"
}

# ═══════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cancer-variant-api")

# ═══════════════════════════════════════════════════════════════
# IN-MEMORY JOB STORAGE
# ═══════════════════════════════════════════════════════════════
jobs_db: Dict[str, Dict] = {}
MAX_JOBS = 100  # Prevent memory leak

def cleanup_old_jobs():
    """Remove completed/failed jobs when storage exceeds limit"""
    if len(jobs_db) > MAX_JOBS:
        completed_jobs = [
            jid for jid, job in jobs_db.items() 
            if job.get("status") in ["completed", "failed"]
        ]
        for jid in completed_jobs[:len(completed_jobs) - MAX_JOBS // 2]:
            del jobs_db[jid]
        logger.info(f"Cleaned up old jobs. Remaining: {len(jobs_db)}")

# ═══════════════════════════════════════════════════════════════
# WEBSOCKET & PROGRESS MANAGEMENT
# ═══════════════════════════════════════════════════════════════
class ConnectionManager:
    """Thread-safe WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)
        logger.info(f"✅ WebSocket connected. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            self.active_connections.discard(websocket)
        logger.info(f"❌ WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            logger.debug(f"No active connections to broadcast: {message}")
            return
        
        async with self.lock:
            connections_snapshot = list(self.active_connections)
        
        failed_connections = []
        for connection in connections_snapshot:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                failed_connections.append(connection)
        
        # Clean up failed connections
        if failed_connections:
            async with self.lock:
                for conn in failed_connections:
                    self.active_connections.discard(conn)

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP SETUP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="Cancer Variant Analysis API",
    description="Deterministic cancer variant analysis with LLM interpretation",
    version="2.0.0",
)

# CORS Middleware - MUST be first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# ═══════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time progress updates via WebSocket"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"WS message: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await manager.disconnect(websocket)

# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    """Health check endpoint"""
    processing_count = sum(
        1 for j in jobs_db.values() 
        if j.get("status") == "processing"
    )
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "jobs_total": len(jobs_db),
        "jobs_processing": processing_count
    }

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def safe_float(val, default=0.0):
    """Safely convert value to float"""
    if val is None or pd.isna(val) or str(val).strip() == ".":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=1):
    """Safely convert value to int"""
    if val is None or pd.isna(val) or str(val).strip() == ".":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def find_column(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    """Find column by case-insensitive search"""
    df_cols_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in df_cols_lower:
            return df_cols_lower[candidate.lower()]
    return default

def has_annotation_columns(df: pd.DataFrame) -> bool:
    """Check if dataframe has ANNOVAR annotation columns"""
    return bool(REQUIRED_ANNOVAR_COLS & set(df.columns))

def is_raw_vcf(df: pd.DataFrame) -> bool:
    """Check if dataframe is raw VCF without annotations"""
    vcf_cols = {"CHROM", "POS", "REF", "ALT", "QUAL", "FILTER"}
    return vcf_cols.issubset(df.columns) and not has_annotation_columns(df)

def create_minimal_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """Create minimal annotation for raw VCF files"""
    df = df.copy()
    rename_map = {"CHROM": "Chr", "POS": "Start", "REF": "Ref", "ALT": "Alt"}
    df = df.rename(columns=rename_map)
    placeholder_cols = {
        "Gene.refGene": "UNKNOWN",
        "Gene.refGeneWithVer": "UNKNOWN",
        "ExonicFunc.refGene": "unknown",
        "ExonicFunc.refGeneWithVer": "unknown",
        "AAChange.refGene": ".",
        "AAChange.refGeneWithVer": ".",
    }
    for col, value in placeholder_cols.items():
        if col not in df.columns:
            df[col] = value
    return df

# ═══════════════════════════════════════════════════════════════
# FILE PARSING
# ═══════════════════════════════════════════════════════════════
async def parse_vcf(file_path: str) -> pd.DataFrame:
    """Parse VCF/VCF.GZ file"""
    try:
        logger.info(f"Parsing VCF: {file_path}")
        vcf = pysam.VariantFile(file_path)
        records = []
        
        for rec in vcf.fetch():
            record = {
                "CHROM": rec.chrom,
                "POS": rec.pos,
                "REF": rec.ref,
                "ALT": ",".join(map(str, rec.alts or [])),
                "QUAL": rec.qual,
                "FILTER": ";".join(rec.filter.keys()) if rec.filter else "PASS"
            }
            # Add INFO fields
            for k, v in rec.info.items():
                record[k] = ",".join(map(str, v)) if isinstance(v, tuple) else v
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Parsed {len(df)} variants from VCF")
        return df
        
    except Exception as e:
        logger.error(f"VCF parsing failed: {e}", exc_info=True)
        raise HTTPException(400, f"VCF parsing failed: {str(e)}")

async def parse_table(file_path: str, ext: str) -> pd.DataFrame:
    """Parse CSV/TSV/TXT file"""
    try:
        logger.info(f"Parsing table ({ext}): {file_path}")
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(file_path, sep=sep, dtype={"CHROM": str, "Chr": str}, low_memory=False)
        
        # Normalize column names
        df = df.rename(columns={
            "CHROMOSOME": "CHROM", "CHR": "CHROM",
            "START": "POS", "POSITION": "POS",
            "REFERENCE": "REF", "ALTERNATE": "ALT"
        })
        
        logger.info(f"Parsed {len(df)} variants from table")
        return df
        
    except Exception as e:
        logger.error(f"Table parsing failed: {e}", exc_info=True)
        raise HTTPException(400, f"Table parsing failed: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# VARIANT ANALYSIS
# ═══════════════════════════════════════════════════════════════
def clean_variants_for_output(df: pd.DataFrame) -> List[Dict]:
    """Convert dataframe to clean variant objects"""
    cleaned = []
    
    # Find key columns
    chr_col = find_column(df, ["chr", "chrom", "chromosome"])
    pos_col = find_column(df, ["start", "pos", "position"])
    ref_col = find_column(df, ["ref", "reference"])
    alt_col = find_column(df, ["alt", "alternate"])
    
    if not all([chr_col, pos_col, ref_col, alt_col]):
        logger.warning("Missing required coordinate columns")
        return []
    
    for idx, row in df.iterrows():
        try:
            chr_val = row.get(chr_col)
            pos_val = row.get(pos_col)
            
            if pd.isna(chr_val) or pd.isna(pos_val):
                continue
            
            gene_col = find_column(df, ["gene.refgenewithver", "gene.refgene", "gene"], "UNKNOWN")
            effect_col = find_column(df, ["exonicfunc.refgenewithver", "exonicfunc.refgene"], "unknown")
            aachange_col = find_column(df, ["aachange.refgenewithver", "aachange.refgene"], "UNKNOWN:p.?")
            clnsig_col = find_column(df, ["clnsig"], "NA")
            
            variant_obj = {
                "chrom": str(chr_val),
                "pos": safe_int(pos_val),
                "ref": str(row.get(ref_col, "NA")),
                "alt": str(row.get(alt_col, "NA")),
                "DP": safe_int(row.get("DP"), 1),
                "AF": safe_float(row.get("AF"), 0.01),
                "gene": str(row.get(gene_col, "UNKNOWN")) if gene_col else "UNKNOWN",
                "effect": str(row.get(effect_col, "unknown")) if effect_col else "unknown",
                "aachange": str(row.get(aachange_col, "UNKNOWN:p.?")) if aachange_col else "UNKNOWN:p.?",
                "clnsig": str(row.get(clnsig_col, "NA")) if clnsig_col else "NA",
            }
            cleaned.append(variant_obj)
        except Exception as e:
            logger.debug(f"Error processing variant row {idx}: {e}")
            continue
    
    logger.info(f"Cleaned {len(cleaned)} variants for output")
    return cleaned

async def analyze_wilms_tumor(filtered_df: pd.DataFrame, annotated_df: pd.DataFrame) -> Dict:
    """Analyze Wilms tumor associated variants"""
    try:
        logger.info("Starting Wilms tumor analysis")
        
        wilms_genes = load_wilms_panel()
        wilms_df = annotated_df[annotated_df.get("IS_WILMS_GENE", False)].copy()
        
        if wilms_df.empty:
            logger.info("No Wilms tumor variants detected")
            return {
                "wilms_variant_count": 0,
                "wilms_gene_burden": [],
                "literature_summary": "No Wilms tumor–associated variants detected."
            }
        
        # Prepare Wilms variants
        wilms_cols_needed = ["CHROM", "POS", "REF", "ALT"]
        wilms_df_valid = wilms_df.dropna(subset=wilms_cols_needed)
        
        if wilms_df_valid.empty:
            return {
                "wilms_variant_count": 0,
                "wilms_gene_burden": [],
                "literature_summary": "No valid Wilms variants"
            }
        
        # Construct payload
        wilms_payload = wilms_df_valid[wilms_cols_needed].to_dict(orient="records")
        
        # Get literature reasoning
        lit_summary = wilms_literature_reasoning(wilms_payload)
        
        # Calculate burden
        burden = wilms_gene_burden(annotated_df).to_dict(orient="records")
        
        return {
            "wilms_variant_count": len(wilms_df),
            "wilms_gene_burden": burden,
            "literature_summary": lit_summary
        }
        
    except Exception as e:
        logger.error(f"Wilms analysis failed: {e}", exc_info=True)
        return {
            "wilms_variant_count": 0,
            "wilms_gene_burden": [],
            "literature_summary": f"Wilms analysis failed: {str(e)}"
        }

# ═══════════════════════════════════════════════════════════════
# BACKGROUND TASK
# ═══════════════════════════════════════════════════════════════
async def run_analysis_task(
    job_id: str,
    file_path: str,
    ext: str,
    prompt: str,
    disease: str,
    max_lit_variants_int: int,
    original_df_len: int
):
    """Background task for variant analysis"""
    try:
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["started_at"] = datetime.now().isoformat()
        
        await manager.broadcast(f"Job {job_id}: Starting analysis...")
        logger.info(f"Job {job_id}: Analysis started")
        
        # Parse file
        await manager.broadcast(f"Job {job_id}: Parsing file...")
        if ext in (".vcf", ".vcf.gz"):
            df = await parse_vcf(file_path)
        else:
            df = await parse_table(file_path, ext)
        
        original_count = len(df)
        jobs_db[job_id]["variants_input"] = original_count
        
        # Filter variants
        await manager.broadcast(f"Job {job_id}: Filtering variants...")
        filtered = filter_variants(df)
        
        if filtered.empty:
            jobs_db[job_id] = {
                "status": "completed",
                "variants_input": original_count,
                "variants_filtered": 0,
                "results": {
                    "summary": "No clinically relevant variants found.",
                    "variants": []
                },
                "completed_at": datetime.now().isoformat()
            }
            await manager.broadcast(f"Job {job_id}: Complete - no relevant variants")
            logger.info(f"Job {job_id}: Completed - no relevant variants")
            return
        
        jobs_db[job_id]["variants_filtered"] = len(filtered)
        
        # Analyze variants
        await manager.broadcast(f"Job {job_id}: Analyzing {len(filtered)} variants...")
        analysis = analyze_variants(
            filtered,
            prompt,
            disease=disease,
            max_lit_variants=max_lit_variants_int
        )
        
        # Wilms tumor analysis
        await manager.broadcast(f"Job {job_id}: Performing Wilms tumor analysis...")
        wilms_genes = load_wilms_panel()
        annotated = annotate_wilms_burden(filtered, wilms_genes)
        wilms_result = await analyze_wilms_tumor(filtered, annotated)
        
        # Clean output
        await manager.broadcast(f"Job {job_id}: Formatting results...")
        cleaned_variants = clean_variants_for_output(filtered)
        
        # Store results
        jobs_db[job_id] = {
            "status": "completed",
            "variants_input": original_count,
            "variants_filtered": len(filtered),
            "results": {
                "summary": analysis.get("summary", "Analysis complete"),
                "variants": cleaned_variants,
                "wilms": wilms_result,
            },
            "completed_at": datetime.now().isoformat()
        }
        
        await manager.broadcast(f"Job {job_id}: ✅ Analysis complete!")
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs_db[job_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "failed_at": datetime.now().isoformat()
        }
        await manager.broadcast(f"Job {job_id}: ❌ Error - {str(e)}")
    
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file: {e}")
        
        # Cleanup old jobs
        cleanup_old_jobs()

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    disease: str = Form("Unknown"),
    max_lit_variants: str = Form("300"),
):
    """
    Submit variant analysis job - RETURNS IMMEDIATELY
    
    Analysis runs asynchronously in background. Frontend should:
    1. Get job_id from response
    2. Connect to WebSocket /ws/progress for live updates
    3. Poll /results/{job_id} to check status and get results
    
    Parameters:
    - file: VCF, VCF.GZ, CSV, TSV, or TXT file
    - prompt: Custom analysis prompt
    - disease: Disease type (e.g., "Wilms", "Unknown")
    - max_lit_variants: Maximum variants to analyze (default: 300)
    
    Response: 202 Accepted - Job submitted successfully
    """
    job_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Job {job_id}: New analysis request - {file.filename}")
        
        # Validate filename
        if not file.filename:
            raise HTTPException(400, "No filename provided")
        
        # Validate extension
        ext = next(
            (e for e in VALID_EXTENSIONS if file.filename.lower().endswith(e)),
            None
        )
        if not ext:
            raise HTTPException(
                400,
                f"Invalid file type. Supported: {', '.join(VALID_EXTENSIONS)}"
            )
        
        # Parse max_lit_variants
        try:
            max_lit_variants_int = int(max_lit_variants)
            max_lit_variants_int = max(1, min(max_lit_variants_int, 10000))
        except (ValueError, TypeError):
            max_lit_variants_int = 300
        
        # Create job record IMMEDIATELY
        jobs_db[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "disease": disease,
            "max_lit_variants": max_lit_variants_int,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Job {job_id}: Record created")
        
        # Read file
        try:
            content = await asyncio.wait_for(file.read(), timeout=30.0)
            if not content:
                raise HTTPException(400, "File is empty")
            logger.info(f"Job {job_id}: File read successfully ({len(content)} bytes)")
        except asyncio.TimeoutError:
            raise HTTPException(408, "File read timeout")
        except Exception as e:
            logger.error(f"File read error: {e}")
            raise HTTPException(400, f"Failed to read file: {str(e)}")
        
        # Create temp file
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(content)
            tmp.close()
            logger.info(f"Job {job_id}: Temp file created at {tmp.name}")
        except Exception as e:
            logger.error(f"Temp file creation failed: {e}")
            jobs_db[job_id]["status"] = "error"
            jobs_db[job_id]["error"] = f"Failed to create temp file: {str(e)}"
            raise HTTPException(500, f"Server error: {str(e)}")
        
        # QUEUE BACKGROUND TASK - DON'T WAIT FOR IT
        background_tasks.add_task(
            run_analysis_task,
            job_id=job_id,
            file_path=tmp.name,
            ext=ext,
            prompt=prompt,
            disease=disease,
            max_lit_variants_int=max_lit_variants_int,
            original_df_len=0
        )
        
        logger.info(f"Job {job_id}: Queued for background processing - RETURNING JOB_ID NOW")
        await manager.broadcast(f"Job {job_id}: Analysis job submitted")
        
        # RETURN IMMEDIATELY WITH JOB_ID (Status 202 = Accepted)
        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "filename": file.filename,
                "message": "Analysis started. Use job_id to poll /results/{job_id}",
                "next_steps": [
                    f"1. Poll GET /results/{job_id} every 2-3 seconds",
                    "2. Connect WebSocket to /ws/progress for live updates",
                    "3. When status='completed', results will be in response"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        if job_id in jobs_db:
            jobs_db[job_id]["status"] = "error"
            jobs_db[job_id]["error"] = str(e)
        raise HTTPException(500, f"Server error: {str(e)}")

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results for a job"""
    if job_id not in jobs_db:
        raise HTTPException(404, f"Job {job_id} not found")
    
    job = jobs_db[job_id]
    
    # Don't return traceback to frontend
    if "traceback" in job:
        job_copy = job.copy()
        del job_copy["traceback"]
        return job_copy
    
    return job

@app.get("/jobs")
async def list_jobs():
    """List all jobs with their status"""
    return {
        "total": len(jobs_db),
        "jobs": {
            job_id: {
                "status": job.get("status"),
                "filename": job.get("filename"),
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at")
            }
            for job_id, job in jobs_db.items()
        }
    }

# ═══════════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN
# ═══════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    logger.info("🚀 Cancer Variant Analysis API starting...")
    logger.info(f"CORS enabled for all origins")

@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 API shutting down...")
    jobs_db.clear()

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )