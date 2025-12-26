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
            loop = self.loop or asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.broadcast_func(msg))
        except Exception:
            pass


class ConnectionManager:
    """Manages WebSocket connections for real-time progress"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                self.active_connections.discard(connection)


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


@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Configure logging
va_logger = logging.getLogger("variant-agents")
va_logger.propagate = False
progress_handler = ProgressLogger(manager.broadcast)
progress_handler.setFormatter(logging.Formatter('%(message)s'))
va_logger.addHandler(progress_handler)

wes_logger = logging.getLogger("wes-agent")
wes_logger.addHandler(progress_handler)

logger.info("Loggers configured with WebSocket progress tracking")


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
        logger.info(f"Parsed {len(df)} variants from VCF")
        return df
    
    except Exception as e:
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
        
        logger.info(f"Parsed {len(df)} variants from table")
        return df
    
    except Exception as e:
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
        logger.info(f"Analyzed {len(wilms_payload)} Wilms tumor variants")
        
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
# Variant Cleaning
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
    
    logger.info(f"Successfully formatted {len(cleaned)} variants for output")
    
    return cleaned


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "Cancer Variant Analysis API", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
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
    """
    
    logger.info(f"Analysis request: {file.filename} | Disease: {disease}")
    
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
        
        logger.info(f"File saved: {file_path}")
        
        # Parse file
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
        
        logger.info(f"Parsed {len(df)} variants")
        
        # Handle unannotated VCF
        if is_raw_vcf(df):
            logger.warning("Input is unannotated VCF")
            df = create_minimal_annotation(df)
        
        # Filter variants
        logger.info("Starting variant filtering...")
        try:
            filtered = filter_variants(df)
            logger.info(f"Filtering complete: {len(filtered)} variants passed")
        except ValueError as e:
            logger.error(f"Filtering failed: {e}")
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
        logger.info(f"Starting variant analysis (disease={disease})...")
        try:
            analysis = analyze_variants(
                filtered, prompt, disease=disease, max_lit_variants=max_lit_variants_int
            )
            logger.info("Analysis complete")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise HTTPException(500, f"Variant analysis failed: {str(e)}")
        
        # Wilms tumor analysis
        logger.info("Starting Wilms tumor analysis...")
        try:
            wilms_genes = load_wilms_panel()
            annotated = annotate_wilms_burden(filtered, wilms_genes)
            wilms_result = await analyze_wilms_tumor(filtered, annotated)
        except Exception as e:
            logger.warning(f"Wilms analysis skipped: {e}")
            wilms_result = {
                "wilms_variant_count": 0,
                "wilms_gene_burden": [],
                "literature_summary": "Wilms analysis unavailable",
            }
        
        # Clean output
        cleaned_variants = clean_variants_for_output(filtered)
        
        # Return response
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
        logger.exception("Analysis failed")
        raise HTTPException(500, f"Analysis failed: {str(e)}")