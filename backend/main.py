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

# -------------------------------------------------
# Async Job Storage
# -------------------------------------------------
# For production use a real DB or Redis. For local, this works.
jobs_db: Dict[str, Dict] = {}

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
    "*",
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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = self.loop
            
            if loop and loop.is_running():
                asyncio.create_task(self.broadcast_func(msg))
        except Exception:
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
        if not self.active_connections:
            return
        
        connections_snapshot = list(self.active_connections)
        failed_connections = []
        
        for connection in connections_snapshot:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                failed_connections.append(connection)
        
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# -------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
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


# -------------------------------------------------
# Helper Functions: Annotation Validation & Cleaning
# -------------------------------------------------
def has_annotation_columns(df: pd.DataFrame) -> bool:
    return bool(REQUIRED_ANNOVAR_COLS & set(df.columns))

def is_raw_vcf(df: pd.DataFrame) -> bool:
    vcf_cols = {"CHROM", "POS", "REF", "ALT", "QUAL", "FILTER"}
    return vcf_cols.issubset(df.columns) and not has_annotation_columns(df)

def create_minimal_annotation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {"CHROM": "Chr", "POS": "Start", "REF": "Ref", "ALT": "Alt"}
    df = df.rename(columns=rename_map)
    placeholder_cols = {
        "Gene.refGene": "UNKNOWN", "Gene.refGeneWithVer": "UNKNOWN",
        "ExonicFunc.refGene": "unknown", "ExonicFunc.refGeneWithVer": "unknown",
        "AAChange.refGene": ".", "AAChange.refGeneWithVer": ".",
    }
    for col, value in placeholder_cols.items():
        if col not in df.columns: df[col] = value
    return df

def safe_get_wilms_columns(df: pd.DataFrame) -> Tuple[Dict[str, Optional[str]], pd.DataFrame]:
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
                break
        result_mapping[desired_name] = found
    return result_mapping, df_normalized

async def parse_vcf(file_path: str) -> pd.DataFrame:
    try:
        vcf = pysam.VariantFile(file_path)
        records = []
        for rec in vcf.fetch():
            record = {"CHROM": rec.chrom, "POS": rec.pos, "REF": rec.ref, "ALT": ",".join(map(str, rec.alts or [])), "QUAL": rec.qual, "FILTER": ";".join(rec.filter.keys()) if rec.filter else "PASS"}
            for k, v in rec.info.items(): record[k] = ",".join(map(str, v)) if isinstance(v, tuple) else v
            records.append(record)
        return pd.DataFrame(records)
    except Exception as e: raise HTTPException(400, f"VCF parsing failed: {str(e)}")

async def parse_table(file_path: str, ext: str) -> pd.DataFrame:
    try:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(file_path, sep=sep, dtype={"CHROM": str, "Chr": str})
        df = df.rename(columns={"CHROMOSOME": "CHROM", "CHR": "CHROM", "START": "POS", "POSITION": "POS", "REFERENCE": "REF", "ALTERNATE": "ALT"})
        return df
    except Exception as e: raise HTTPException(400, f"Table parsing failed: {str(e)}")

async def analyze_wilms_tumor(filtered_df: pd.DataFrame, annotated_df: pd.DataFrame) -> Dict:
    try:
        wilms_genes = load_wilms_panel()
        wilms_df = annotated_df[annotated_df.get("IS_WILMS_GENE", False)].copy()
        if wilms_df.empty: return {"wilms_variant_count": 0, "wilms_gene_burden": [], "literature_summary": "No Wilms tumor–associated variants detected."}
        col_mapping, wilms_df_normalized = safe_get_wilms_columns(wilms_df)
        required_cols = {"chrom", "pos", "ref", "alt"}
        if not required_cols.issubset(col_mapping) or None in [col_mapping.get(k) for k in required_cols]: return {"wilms_variant_count": 0, "wilms_gene_burden": [], "literature_summary": "Wilms analysis unavailable: missing coordinates"}
        wilms_cols = [col_mapping[k] for k in required_cols]
        for key in ("gene", "effect", "clnsig"):
            if col_mapping.get(key): wilms_cols.append(col_mapping[key])
        wilms_df_valid = wilms_df_normalized.dropna(subset=[col_mapping[k] for k in required_cols])
        if wilms_df_valid.empty: return {"wilms_variant_count": 0, "wilms_gene_burden": [], "literature_summary": "No valid Wilms variants"}
        wilms_payload = wilms_df_valid[wilms_cols].to_dict(orient="records")
        lit_summary = wilms_literature_reasoning(wilms_payload)
        burden = wilms_gene_burden(annotated_df).to_dict(orient="records")
        return {"wilms_variant_count": len(wilms_df), "wilms_gene_burden": burden, "literature_summary": lit_summary}
    except Exception as e: return {"wilms_variant_count": 0, "wilms_gene_burden": [], "literature_summary": f"Wilms analysis failed: {e}"}

def safe_float(val, default=0):
    if val is None or pd.isna(val) or str(val).strip() == ".": return default
    try: return float(val)
    except: return default

def safe_int(val, default=1):
    if val is None or pd.isna(val) or str(val).strip() == ".": return default
    try: return int(float(val))
    except: return default

def clean_variants_for_output(df: pd.DataFrame) -> List[Dict]:
    cleaned = []
    df_cols_lower = {col.lower(): col for col in df.columns}
    chr_col = next((df_cols_lower.get(k) for k in ["chr", "chrom", "chromosome"]), None)
    pos_col = next((df_cols_lower.get(k) for k in ["start", "pos", "position"]), None)
    ref_col = next((df_cols_lower.get(k) for k in ["ref", "reference"]), None)
    alt_col = next((df_cols_lower.get(k) for k in ["alt", "alternate"]), None)
    if not all([chr_col, pos_col, ref_col, alt_col]): return []
    for idx, row in df.iterrows():
        try:
            chr_val, start_val = row.get(chr_col), row.get(pos_col)
            if pd.isna(chr_val) or pd.isna(start_val): continue
            variant_obj = {
                "chrom": str(chr_val), "pos": int(start_val), "ref": str(row.get(ref_col, "NA")), "alt": str(row.get(alt_col, "NA")),
                "DP": safe_int(row.get("DP"), 1), "AF": safe_float(row.get("AF"), 0.01),
                "Gene.refGeneWithVer": str(next((row.get(col) for col in df.columns if col.lower() in ["gene.refgenewithver", "gene.refgene"]), "UNKNOWN")),
                "ExonicFunc.refGeneWithVer": str(next((row.get(col) for col in df.columns if col.lower() in ["exonicfunc.refgenewithver", "exonicfunc.refgene"]), "unknown")),
                "AAChange.refGeneWithVer": str(next((row.get(col) for col in df.columns if col.lower() in ["aachange.refgenewithver", "aachange.refgene"]), "UNKNOWN:p.?")),
                "CLNSIG": str(next((row.get(col) for col in df.columns if col.lower() == "clnsig"), "NA")),
            }
            cleaned.append(variant_obj)
        except: continue
    return cleaned

# -------------------------------------------------
# Async Job Management
# -------------------------------------------------
async def run_analysis_task(job_id: str, file_path: str, ext: str, prompt: str, disease: str, max_lit_variants_int: int, original_df_len: int):
    try:
        jobs_db[job_id]["status"] = "processing"
        await manager.broadcast(f"Job {job_id}: Processing started")
        if ext in (".vcf", ".vcf.gz"): df = await parse_vcf(file_path)
        else: df = await parse_table(file_path, ext)
        filtered = filter_variants(df)
        if filtered.empty:
            jobs_db[job_id] = {"status": "completed", "results": {"input_variants": original_df_len, "filtered_variants": 0, "results": {"summary": "No clinically relevant variants found.", "variants": []}}}
            return
        analysis = analyze_variants(filtered, prompt, disease=disease, max_lit_variants=max_lit_variants_int)
        wilms_genes = load_wilms_panel()
        annotated = annotate_wilms_burden(filtered, wilms_genes)
        wilms_result = await analyze_wilms_tumor(filtered, annotated)
        cleaned_variants = clean_variants_for_output(filtered)
        jobs_db[job_id] = {
            "status": "completed",
            "results": {
                "input_variants": original_df_len, "filtered_variants": len(filtered), "disease": disease,
                "results": {
                    "summary": analysis.get("summary", "Analysis complete"), "variant_count": len(filtered),
                    "variants": cleaned_variants, "wilms": wilms_result,
                }
            }
        }
        await manager.broadcast(f"Job {job_id}: Analysis complete!")
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs_db[job_id] = {"status": "error", "error": str(e)}
        await manager.broadcast(f"Job {job_id}: Error - {str(e)}")

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    disease: str = Form("Unknown"),
    max_lit_variants: str = Form("300"),
):
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "queued", "filename": file.filename}
    
    try:
        content = await file.read()
        ext = next((e for e in VALID_EXTENSIONS if file.filename.lower().endswith(e)), ".txt")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(content)
        tmp.close()
        
        background_tasks.add_task(
            run_analysis_task, job_id, tmp.name, ext, prompt, disease, int(max_lit_variants), 0
        )
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        jobs_db[job_id] = {"status": "error", "error": str(e)}
        return {"job_id": job_id, "status": "error", "error": str(e)}

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs_db: raise HTTPException(404, "Job not found")
    return jobs_db[job_id]

@app.get("/health")
def health(): return {"status": "healthy", "jobs_active": len([j for j in jobs_db.values() if j["status"] == "processing"])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
