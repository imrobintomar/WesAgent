"""
FastAPI Backend Server for Cancer Variant Analysis
Deterministic variant pipeline + Ollama clinical reasoning
VCF / VCF.GZ compatible with graceful handling
"""

import json
import asyncio
import logging
import tempfile
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple

import pandas as pd
import pysam
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from services.franklin_client import parse_variant, classify_variant as franklin_classify
from services.varsome_client import classify_variant as varsome_classify

from variant_agents import (
    filter_variants,
    analyze_variants,
    wilms_literature_reasoning,
    normalize_annovar_columns,
)
from panels.wilms_panel import load_wilms_panel
from burden_engine import annotate_wilms_burden, wilms_gene_burden

# -------------------------------------------------
# Configuration & Setup
# -------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wes-agent")

VALID_EXTENSIONS = (".vcf", ".vcf.gz", ".txt", ".tsv", ".csv")
REQUIRED_ANNOVAR_COLS = {
    "Gene.refGene", "Gene.refGeneWithVer",
    "ExonicFunc.refGene", "ExonicFunc.refGeneWithVer",
    "AAChange.refGene", "AAChange.refGeneWithVer"
}

# CORS Origins - Configure for your deployment
CORS_ORIGINS = [
    "https://wes-agent.vercel.app",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]


# -------------------------------------------------
# Progress Logging via WebSocket
# -------------------------------------------------
class ProgressLogger(logging.Handler):
    """Custom handler to broadcast logs via WebSocket"""
    
    def __init__(self, broadcast_func):
        super().__init__()
        self.broadcast_func = broadcast_func
        self.loop = None

    def set_loop(self, loop):
        self.loop = loop

    def emit(self, record):
        msg = self.format(record)
        try:
            # Try to get the running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self.loop
            
            if loop and loop.is_running():
                asyncio.create_task(self.broadcast_func(msg))
        except Exception as e:
            # Silently fail if we can't broadcast (don't disrupt logging)
            pass


class ConnectionManager:
    """Manages WebSocket connections for real-time progress - THREAD-SAFE"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """
        FIXED: Create snapshot of connections to avoid 'Set changed size during iteration'
        This was causing RuntimeError in your logs.
        """
        if not self.active_connections:
            return
        
        # Critical fix: Create a list copy BEFORE iterating
        connections_snapshot = list(self.active_connections)
        
        # Track failed connections for cleanup
        failed_connections = []
        
        for connection in connections_snapshot:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                failed_connections.append(connection)
        
        # Clean up failed connections
        for conn in failed_connections:
            self.active_connections.discard(conn)


# -------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------
app = FastAPI(
    title="Cancer Variant Analysis API",
    description="Deterministic cancer variant analysis with LLM interpretation",
    version="1.0.0",
)

# FIXED CORS Configuration - Must be added BEFORE routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Specific origins instead of "*" for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

manager = ConnectionManager()


# -------------------------------------------------
# CORS Preflight Handler & Error Handler
# -------------------------------------------------
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """
    Handle CORS preflight (OPTIONS) requests.
    This is critical for browser-based file uploads.
    """
    return JSONResponse(status_code=200)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler that includes CORS headers.
    Ensures 503 and other errors still have CORS headers.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTP exception handler that includes CORS headers.
    """
    logger.error(f"HTTP exception {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
        }
    )


# -------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive - wait for client messages
            data = await websocket.receive_text()
            logger.debug(f"WS message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Configure logging with progress broadcasting
va_logger = logging.getLogger("variant-agents")
va_logger.propagate = False
progress_handler = ProgressLogger(manager.broadcast)
progress_handler.setFormatter(logging.Formatter('%(message)s'))
va_logger.addHandler(progress_handler)

wes_logger = logging.getLogger("wes-agent")
wes_logger.addHandler(progress_handler)

logger.info("✅ Loggers configured with WebSocket progress tracking")
logger.info(f"✅ CORS configured for origins: {CORS_ORIGINS}")


# -------------------------------------------------
# Helper Functions: Annotation Validation
# -------------------------------------------------
def has_annotation_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame contains ANNOVAR annotation columns"""
    return bool(REQUIRED_ANNOVAR_COLS & set(df.columns))


def is_raw_vcf(df: pd.DataFrame) -> bool:
    """Detect if this is an unannotated raw VCF"""
    vcf_cols = {"CHROM", "POS", "REF", "ALT", "QUAL", "FILTER"}
    return vcf_cols.issubset(df.columns) and not has_annotation_columns(df)


def create_minimal_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create minimal annotation columns for unannotated VCF files.
    Allows pipeline to run but with reduced biological interpretation.
    """
    df = df.copy()
    
    # Standardize column names
    rename_map = {"CHROM": "Chr", "POS": "Start", "REF": "Ref", "ALT": "Alt"}
    df = df.rename(columns=rename_map)
    
    # Add placeholder annotation columns
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
    
    logger.warning("⚠️  Using minimal annotation. Results will be limited. "
                   "Please use ANNOVAR-annotated VCF for best results.")
    return df


def safe_get_wilms_columns(df: pd.DataFrame) -> Tuple[Dict[str, Optional[str]], pd.DataFrame]:
    """
    Safely resolve Wilms tumor column names with fallbacks.
    
    Returns:
        Tuple of (column_mapping_dict, normalized_df)
    """
    df_normalized = normalize_annovar_columns(df)
    
    column_mapping = {
        "gene": ["gene.refgene", "gene_symbol", "genename", "Gene.refGene"],
        "effect": ["exonicfunc.refgene", "variant_effect", "ExonicFunc.refGene"],
        "clnsig": ["clnsig", "clinical_significance", "CLNSIG"],
        "chrom": ["chrom", "chromosome", "chr", "Chr"],
        "pos": ["pos", "position", "start", "Start"],
        "ref": ["ref", "reference"],
        "alt": ["alt", "alternate"],
    }
    
    result_mapping = {}
    df_cols_lower = {col.lower(): col for col in df_normalized.columns}
    
    for desired_name, possible_names in column_mapping.items():
        found = None
        for name in possible_names:
            if name.lower() in df_cols_lower:
                found = df_cols_lower[name.lower()]
                logger.info(f"Mapped '{desired_name}' -> '{found}'")
                break
        
        result_mapping[desired_name] = found
        if not found:
            logger.warning(f"Could not find column for '{desired_name}'")
    
    return result_mapping, df_normalized


# -------------------------------------------------
# File Parsing
# -------------------------------------------------
async def parse_vcf(file_path: str) -> pd.DataFrame:
    """Parse VCF or VCF.GZ file into DataFrame"""
    try:
        vcf = pysam.VariantFile(file_path)
        records = []
        
        for rec in vcf.fetch():
            record = {
                "CHROM": rec.chrom,
                "POS": rec.pos,
                "REF": rec.ref,
                "ALT": ",".join(map(str, rec.alts or [])),
                "QUAL": rec.qual,
                "FILTER": ";".join(rec.filter.keys()) if rec.filter else "PASS",
            }
            # Include INFO fields
            for k, v in rec.info.items():
                record[k] = ",".join(map(str, v)) if isinstance(v, tuple) else v
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"✅ Parsed {len(df)} variants from VCF")
        return df
    
    except Exception as e:
        logger.error(f"VCF parsing error: {str(e)}")
        raise HTTPException(400, f"VCF parsing failed: {str(e)}")


async def parse_table(file_path: str, ext: str) -> pd.DataFrame:
    """Parse CSV/TSV/TXT file into DataFrame"""
    try:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(file_path, sep=sep, dtype={"CHROM": str, "Chr": str})
        
        # Standardize common column names
        rename_map = {
            "CHROMOSOME": "CHROM", "CHR": "CHROM",
            "START": "POS", "POSITION": "POS",
            "REFERENCE": "REF", "ALTERNATE": "ALT",
        }
        df = df.rename(columns=rename_map)
        
        logger.info(f"✅ Parsed {len(df)} variants from table")
        return df
    
    except Exception as e:
        logger.error(f"Table parsing error: {str(e)}")
        raise HTTPException(400, f"Table parsing failed: {str(e)}")


# -------------------------------------------------
# Wilms Tumor Analysis
# -------------------------------------------------
async def analyze_wilms_tumor(
    filtered_df: pd.DataFrame, annotated_df: pd.DataFrame
) -> Dict:
    """
    Perform Wilms tumor-specific analysis.
    
    Returns:
        Dict with wilms_variant_count, wilms_gene_burden, literature_summary
    """
    try:
        wilms_genes = load_wilms_panel()
        wilms_df = annotated_df[annotated_df.get("IS_WILMS_GENE", False)].copy()
        
        result = {
            "wilms_variant_count": len(wilms_df),
            "wilms_gene_burden": [],
            "literature_summary": "No Wilms tumor–associated variants detected.",
        }
        
        if wilms_df.empty:
            return result
        
        # Get column mapping
        col_mapping, wilms_df_normalized = safe_get_wilms_columns(wilms_df)
        
        # Check for required coordinates
        required_cols = {"chrom", "pos", "ref", "alt"}
        if not required_cols.issubset(col_mapping) or None in [col_mapping.get(k) for k in required_cols]:
            logger.warning("Wilms analysis skipped: missing variant coordinates")
            return result
        
        # Build payload
        wilms_cols = [col_mapping[k] for k in required_cols]
        for key in ("gene", "effect", "clnsig"):
            if col_mapping.get(key):
                wilms_cols.append(col_mapping[key])
        
        wilms_df_valid = wilms_df_normalized.dropna(
            subset=[col_mapping[k] for k in required_cols]
        )
        
        if wilms_df_valid.empty:
            return result
        
        wilms_payload = wilms_df_valid[wilms_cols].to_dict(orient="records")
        result["literature_summary"] = wilms_literature_reasoning(wilms_payload)
        logger.info(f"✅ Analyzed {len(wilms_payload)} Wilms tumor variants")
        
        # Get gene burden
        try:
            result["wilms_gene_burden"] = wilms_gene_burden(annotated_df).to_dict(orient="records")
        except Exception as e:
            logger.warning(f"Could not compute gene burden: {e}")
        
        return result
    
    except Exception as e:
        logger.warning(f"Wilms analysis failed: {e}")
        return {
            "wilms_variant_count": 0,
            "wilms_gene_burden": [],
            "literature_summary": "Wilms analysis unavailable",
        }


# -------------------------------------------------
# Variant Cleaning & Output Formatting
# -------------------------------------------------
def safe_float(val, default=0):
    """Safely convert to float, handling ANNOVAR's '.' and NaN values"""
    if val is None or pd.isna(val) or str(val).strip() == ".":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=1):
    """Safely convert to int, handling ANNOVAR's '.' and NaN values"""
    if val is None or pd.isna(val) or str(val).strip() == ".":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def clean_variants_for_output(df: pd.DataFrame) -> List[Dict]:
    """
    Transform variants to comprehensive format for frontend table.
    Returns fields with proper ANNOVAR naming conventions.
    Handles flexible column naming from different input formats.
    """
    cleaned = []
    
    logger.info(f"clean_variants_for_output: Processing {len(df)} variants")
    logger.info(f"Available DataFrame columns: {df.columns.tolist()}")
    
    # Standardize column names (case-insensitive matching)
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Find the actual column names in the DataFrame
    chr_col = next((df_cols_lower.get(k) for k in ["chr", "chrom", "chromosome"]), None)
    pos_col = next((df_cols_lower.get(k) for k in ["start", "pos", "position"]), None)
    ref_col = next((df_cols_lower.get(k) for k in ["ref", "reference"]), None)
    alt_col = next((df_cols_lower.get(k) for k in ["alt", "alternate"]), None)
    
    logger.info(f"Mapped coordinate columns: chr={chr_col}, pos={pos_col}, ref={ref_col}, alt={alt_col}")
    
    # If we don't have coordinates, we can't proceed
    if not all([chr_col, pos_col, ref_col, alt_col]):
        logger.error(f"Cannot extract coordinates. Available columns: {df.columns.tolist()}")
        return []
    
    for idx, row in df.iterrows():
        try:
            chr_val = row.get(chr_col)
            start_val = row.get(pos_col)
            
            # Skip invalid variants
            if pd.isna(chr_val) or pd.isna(start_val):
                continue
            if pd.isna(row.get(ref_col)) or pd.isna(row.get(alt_col)):
                continue
            
            ref = str(row.get(ref_col, "NA"))
            alt = str(row.get(alt_col, "NA"))
            
            # Search for gene columns - look for exact matches first, then fallbacks
            gene_with_ver = "UNKNOWN"
            aa_change_with_ver = "UNKNOWN:p.?"
            exonic_func_with_ver = "unknown"
            
            # Direct column lookup (case-sensitive first, then insensitive)
            for col in df.columns:
                if col == "Gene.refGeneWithVer" or col.lower() == "gene.refgenewithver":
                    val = row.get(col)
                    if val and val != "." and not pd.isna(val):
                        gene_with_ver = str(val)
                        break
            
            # If not found, try Gene.refGene
            if gene_with_ver == "UNKNOWN":
                for col in df.columns:
                    if col == "Gene.refGene" or col.lower() == "gene.refgene":
                        val = row.get(col)
                        if val and val != "." and not pd.isna(val):
                            gene_with_ver = str(val)
                            break
            
            # Look for AAChange
            for col in df.columns:
                if col == "AAChange.refGeneWithVer" or col.lower() == "aachange.refgenewithver":
                    val = row.get(col)
                    if val and val != "." and not pd.isna(val):
                        aa_change_with_ver = str(val)
                        break
            
            if aa_change_with_ver == "UNKNOWN:p.?":
                for col in df.columns:
                    if col == "AAChange.refGene" or col.lower() == "aachange.refgene":
                        val = row.get(col)
                        if val and val != "." and not pd.isna(val):
                            aa_change_with_ver = str(val)
                            break
            
            # Look for ExonicFunc
            for col in df.columns:
                if col == "ExonicFunc.refGeneWithVer" or col.lower() == "exonicfunc.refgenewithver":
                    val = row.get(col)
                    if val and val != "." and not pd.isna(val):
                        exonic_func_with_ver = str(val)
                        break
            
            if exonic_func_with_ver == "unknown":
                for col in df.columns:
                    if col == "ExonicFunc.refGene" or col.lower() == "exonicfunc.refgene":
                        val = row.get(col)
                        if val and val != "." and not pd.isna(val):
                            exonic_func_with_ver = str(val)
                            break
            
            # Debug: Log first few variants to verify extraction
            if idx < 3:
                logger.info(f"Variant {idx}: {chr_val}:{start_val} | Gene={gene_with_ver} | AAChange={aa_change_with_ver} | ExonicFunc={exonic_func_with_ver}")
            
            variant_obj = {
                # Identifiers (for frontend's buildUniqueVariant fallbacks)
                "chrom": str(chr_val),
                "Chr": str(chr_val),
                "chromosome": str(chr_val),
                "pos": int(start_val),
                "Start": int(start_val),
                "Position": int(start_val),
                "ref": ref,
                "Ref": ref,
                "alt": alt,
                "Alt": alt,
                
                # Depth & Frequency (for filtering)
                "DP": max(1, safe_int(row.get("DP"), 1)),
                "depth": max(1, safe_int(row.get("DP"), 1)),
                "AF": max(0.01, safe_float(row.get("AF"), 0.01)),
                "VAF": max(0.01, safe_float(row.get("VAF"), 0.01)),
                "vaf": max(0.01, safe_float(row.get("VAF"), 0.01)),
                "QUAL": safe_float(row.get("QUAL"), 30),
                
                # Annotation (CRITICAL: must match frontend field names exactly)
                "Gene.refGeneWithVer": str(gene_with_ver),
                "Gene.refGene": str(gene_with_ver),
                "ExonicFunc.refGeneWithVer": str(exonic_func_with_ver),
                "ExonicFunc.refGene": str(exonic_func_with_ver),
                "AAChange.refGeneWithVer": str(aa_change_with_ver),
                "AAChange.refGene": str(aa_change_with_ver),
                
                # Clinical Significance (search for case-insensitive versions)
                "CLNSIG": str(next((row.get(col) for col in df.columns if col.lower() == "clnsig"), "NA")),
                "CLNREV": str(next((row.get(col) for col in df.columns if col.lower() == "clnrev"), "NA")),
                "CLNDBN": str(next((row.get(col) for col in df.columns if col.lower() == "clndbn"), "NA")),
                "CLNVC": str(next((row.get(col) for col in df.columns if col.lower() == "clnvc"), "NA")),
                
                # Prediction Scores
                "SIFT": str(next((row.get(col) for col in df.columns if col.lower() == "sift"), "NA")),
                "PolyPhen2_HDIV_pred": str(next((row.get(col) for col in df.columns if col.lower() == "polyphen2_hdiv_pred"), "NA")),
                "LRT_pred": str(next((row.get(col) for col in df.columns if col.lower() == "lrt_pred"), "NA")),
                "MutationTaster_pred": str(next((row.get(col) for col in df.columns if col.lower() == "mutationtaster_pred"), "NA")),
                
                # Conservation Scores
                "phyloP100way_vertebrate": safe_float(next((row.get(col) for col in df.columns if col.lower() == "phylop100way_vertebrate"), None), 0),
                "GERP++_RS": safe_float(next((row.get(col) for col in df.columns if col.lower() == "gerp++_rs"), None), 0),
                
                # Allele Frequencies
                "ExAC_ALL": safe_float(next((row.get(col) for col in df.columns if col.lower() == "exac_all"), None), 0),
                "gnomAD_ALL": safe_float(next((row.get(col) for col in df.columns if col.lower() == "gnomad_all"), None), 0),
                "1000g2015aug_all": safe_float(next((row.get(col) for col in df.columns if col.lower() == "1000g2015aug_all"), None), 0),
                
                # Additional annotations
                "Func.refGene": str(next((row.get(col) for col in df.columns if col.lower() == "func.refgene"), "NA")),
                "GeneDetail.refGene": str(next((row.get(col) for col in df.columns if col.lower() == "genedetail.refgene"), "NA")),
                "avsnp150": str(next((row.get(col) for col in df.columns if col.lower() == "avsnp150"), "NA")),
                "cosmic": str(next((row.get(col) for col in df.columns if col.lower() == "cosmic"), "NA")),
            }
            
            cleaned.append(variant_obj)
        
        except Exception as e:
            logger.warning(f"Skipped variant row {idx}: {e}")
            continue
    
    logger.info(f"✅ Successfully formatted {len(cleaned)} variants for output")
    
    return cleaned


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Cancer Variant Analysis API",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "websocket_connections": len(manager.active_connections),
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    disease: str = Form("Unknown"),
    max_lit_variants: str = Form("300"),
):
    """
    Analyze variants from uploaded file (VCF, CSV, TSV, or TXT).
    
    Parameters:
        file: VCF/VCF.GZ/CSV/TSV/TXT file
        prompt: Clinical/analysis prompt
        disease: Disease context (default: "Unknown")
        max_lit_variants: Max variants for literature analysis (default: 300)
    
    NOTE: Large files (>100MB) may take several minutes to process.
    Timeouts are set to 15 minutes for analysis completion.
    """
    
    logger.info(f"🔬 Analysis request: {file.filename} | Disease: {disease}")
    await manager.broadcast(f"Analysis started for {file.filename}")
    
    # Validate input
    try:
        max_lit_variants_int = int(max_lit_variants)
    except (ValueError, TypeError):
        max_lit_variants_int = 300
        logger.warning("Invalid max_lit_variants, using default 300")
    
    if not file.filename.lower().endswith(VALID_EXTENSIONS):
        raise HTTPException(400, f"Supported files: {', '.join(VALID_EXTENSIONS)}")
    
    file_path = None
    try:
        # Save uploaded file
        ext = next((e for e in VALID_EXTENSIONS if file.filename.lower().endswith(e)), "")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(400, "Uploaded file is empty")
            tmp.write(content)
            file_path = tmp.name
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"📁 File saved: {file_path} ({file_size_mb:.2f} MB)")
        await manager.broadcast(f"File received ({file_size_mb:.2f} MB) - processing may take several minutes")
        
        # Parse file
        logger.info("📖 Parsing file...")
        await manager.broadcast("Parsing file...")
        if ext in (".vcf", ".vcf.gz"):
            df = await parse_vcf(file_path)
        else:
            df = await parse_table(file_path, ext)
        
        if df.empty:
            return {
                "input_variants": 0,
                "filtered_variants": 0,
                "results": {"summary": "No variants found in file.", "variants": []},
            }
        
        logger.info(f"📊 Parsed {len(df)} variants")
        await manager.broadcast(f"Parsed {len(df)} variants from file")
        
        # Handle unannotated VCF
        if is_raw_vcf(df):
            logger.warning("⚠️ Input is unannotated VCF - using minimal annotation")
            df = create_minimal_annotation(df)
            await manager.broadcast("⚠️ Using minimal annotation for unannotated VCF")
        
        # Filter variants
        logger.info("🔍 Starting variant filtering...")
        await manager.broadcast("Filtering variants...")
        try:
            filtered = filter_variants(df)
            logger.info(f"✅ Filtering complete: {len(filtered)} variants passed")
            await manager.broadcast(f"Filtered to {len(filtered)} variants")
        except ValueError as e:
            logger.error(f"❌ Filtering failed: {e}")
            await manager.broadcast(f"❌ Filtering error: {str(e)}")
            raise HTTPException(
                400,
                f"Filtering error: {str(e)}. Please use ANNOVAR-annotated files for best results."
            )
        
        if filtered.empty:
            return {
                "input_variants": len(df),
                "filtered_variants": 0,
                "results": {"summary": "No clinically relevant variants after filtering.", "variants": []},
            }
        
        # Core variant analysis
        logger.info(f"🧬 Starting variant analysis (disease={disease})...")
        await manager.broadcast(f"Analyzing {len(filtered)} variants for {disease}... (this may take a few minutes)")
        try:
            analysis = analyze_variants(
                filtered, prompt, disease=disease, max_lit_variants=max_lit_variants_int
            )
            logger.info("✅ Analysis complete")
            await manager.broadcast("✅ Variant analysis complete")
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            await manager.broadcast(f"❌ Analysis error: {str(e)}")
            raise HTTPException(500, f"Variant analysis failed: {str(e)}")
        
        # Wilms tumor analysis
        logger.info("🔬 Starting Wilms tumor analysis...")
        await manager.broadcast("Running Wilms tumor analysis...")
        try:
            wilms_genes = load_wilms_panel()
            annotated = annotate_wilms_burden(filtered, wilms_genes)
            wilms_result = await analyze_wilms_tumor(filtered, annotated)
            logger.info("✅ Wilms analysis complete")
            await manager.broadcast("✅ Wilms tumor analysis complete")
        except Exception as e:
            logger.warning(f"⚠️ Wilms analysis skipped: {e}")
            await manager.broadcast(f"⚠️ Wilms analysis skipped")
            wilms_result = {
                "wilms_variant_count": 0,
                "wilms_gene_burden": [],
                "literature_summary": "Wilms analysis unavailable",
            }
        
        # Clean output
        logger.info("📋 Formatting output...")
        await manager.broadcast("Formatting results...")
        cleaned_variants = clean_variants_for_output(filtered)
        
        # Return response
        await manager.broadcast("✅ Analysis complete - generating report")
        
        return {
            "input_variants": len(df),
            "filtered_variants": len(filtered),
            "disease": disease,
            "results": {
                "summary": analysis.get("summary", "Analysis complete"),
                "variant_count": analysis.get("variant_count", 0),
                "prioritized_count": analysis.get("prioritized_count", 0),
                "variants": cleaned_variants,
                "protein_classification": analysis.get("protein_classification", "")[:500],
                "literature_evidence_count": len(analysis.get("literature_evidence", [])),
                "wilms": wilms_result,
            },
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Analysis failed")
        await manager.broadcast(f"❌ Fatal error: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# -------------------------------------------------
# Server Startup & Shutdown
# -------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 WES-Agent Backend Starting...")
    logger.info(f"📍 CORS configured for: {CORS_ORIGINS}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 60)
    logger.info("🛑 WES-Agent Backend Shutting Down")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    # Run with extended timeouts for large file processing
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # CRITICAL: Timeouts for large file handling (in seconds)
        timeout_keep_alive=900,        # 15 minutes for keep-alive connections
        timeout_graceful_shutdown=120,  # 2 minutes graceful shutdown
        # Logging
        access_log=True,
        log_level="info",
        # Increase worker timeout for Gunicorn (if applicable)
        # workers=1,  # Single worker for WebSocket
    )