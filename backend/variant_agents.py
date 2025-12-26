# backend/variant_agents.py

from typing import Dict, List
import pandas as pd
import requests
import json
import logging
from literature.wilms_literature import get_wilms_literature
from panels.wilms_panel import load_wilms_panel

logger = logging.getLogger("variant-agents")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


# -----------------------------
# Deterministic Filtering
# -----------------------------

def normalize_annovar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ANNOVAR column naming across versions (handles case sensitivity)
    Maps both lowercase and UPPERCASE variants to standard names
    """
    
    # Convert column names to lowercase for easier matching
    df.columns = df.columns.str.lower()

    # --- Gene / function fields mapping ---
    gene_map = {
        "gene.refgenewithver": "gene.refgene",
        "func.refgenewithver": "func.refgene",
        "exonicfunc.refgenewithver": "exonicfunc.refgene",
        "aachange.refgenewithver": "aachange.refgene",
    }

    for old, new in gene_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
            logger.info(f"Mapped column: {old} -> {new}")

    # --- gnomAD AF resolution (priority order) ---
    gnomad_candidates = [
        "gnomad41_exome_af_grpmax",
        "gnomad41_exome_af",
        "gnomad41_exome_faf95",
        "gnomad41_exome_faf99",
        "af_exac",
        "af_tgp",
    ]

    for col in gnomad_candidates:
        if col in df.columns:
            # FIX: Use .loc to avoid SettingWithCopyWarning
            df.loc[:, "gnomad_af"] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"Using gnomAD AF from: {col}")
            break
    
    # If no gnomAD column found, create empty one
    if "gnomad_af" not in df.columns:
        df["gnomad_af"] = 0.0
        logger.warning("No gnomAD AF column found, defaulting to 0.0")

    return df


def filter_variants(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_annovar_columns(df)

    # Check for required columns (now lowercase)
    gene_col = "gene.refgene" if "gene.refgene" in df.columns else None
    func_col = "func.refgene" if "func.refgene" in df.columns else None
    exonic_col = "exonicfunc.refgene" if "exonicfunc.refgene" in df.columns else None
    
    missing = []
    if not gene_col:
        missing.append("gene.refgene")
    if not func_col:
        missing.append("func.refgene")
    if not exonic_col:
        missing.append("exonicfunc.refgene")
    
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Available columns: {sorted(df.columns.tolist())}")
        raise ValueError(f"Missing required columns: {missing}")

    # Use found column names for filtering
    filtered = df[
        (df[func_col].isin(["exonic", "splicing"])) &
        (~df[exonic_col].isin(["synonymous SNV", "synonymous snv"])) &
        (pd.to_numeric(df["gnomad_af"], errors="coerce").fillna(0) < 0.01)
    ].copy()

    logger.info(f"Filtered variants: {len(filtered)} from {len(df)} total")
    
    return filtered

# -----------------------------
# ACMG (Simplified)
# -----------------------------
def acmg_classification(row: pd.Series) -> Dict:
    evidence = []

    gnomad_val = row.get("gnomad_af")
    if pd.isna(gnomad_val) or gnomad_val < 0.001:
        evidence.append("PM2")

    exonic_func = row.get("exonicfunc.refgene", "")
    if exonic_func in [
        "nonsynonymous SNV",
        "nonsynonymous snv",
        "frameshift deletion",
        "frameshift insertion",
        "stopgain",
    ]:
        evidence.append("PP3")

    clnsig = row.get("CLNSIG", row.get("clnsig", ""))
    if "Pathogenic" in str(clnsig):
        evidence.append("PS1")

    if "PS1" in evidence and "PM2" in evidence:
        cls = "Pathogenic"
    elif "PM2" in evidence and "PP3" in evidence:
        cls = "Likely Pathogenic"
    else:
        cls = "VUS"

    return {"classification": cls, "evidence": evidence}


# -----------------------------
# Cancer Actionability
# -----------------------------
def cancer_actionability(gene: str) -> Dict:
    db = {
        "BRCA1": ("Tier I", "PARP inhibitors"),
        "BRCA2": ("Tier I", "PARP inhibitors"),
        "EGFR": ("Tier I", "EGFR inhibitors"),
        "ALK": ("Tier I", "ALK inhibitors"),
        "KRAS": ("Tier II", "Emerging therapies"),
        "TP53": ("Tier III", "Prognostic / trials"),
    }

    if gene in db:
        tier, therapy = db[gene]
        return {"actionable": True, "tier": tier, "therapy": therapy}

    return {"actionable": False, "tier": "Tier IV", "therapy": None}


# -----------------------------
# Protein Change Classification
# -----------------------------
def classify_protein_changes(variants: List[Dict]) -> str:
    """
    Use Ollama to classify protein changes from AAChange.refGene field.
    Identifies mutation type and predicted functional impact.
    """
    
    # Extract protein changes
    protein_changes = []
    for variant in variants:
        aa_change = variant.get("AAChange") or variant.get("hgvsp", "")
        gene = variant.get("gene", "")
        if aa_change:
            protein_changes.append({
                "gene": gene,
                "aa_change": aa_change,
                "effect": variant.get("variant_effect", ""),
            })
    
    if not protein_changes:
        return "No protein changes found in variants."
    
    prompt = f"""You are a genomics expert. Classify the following protein changes based on functional impact.

For each protein change, identify:
1. Type of change (missense, frameshift, nonsense, splice site, in-frame indel, etc.)
2. Predicted functional impact (benign, likely benign, uncertain significance, likely pathogenic, pathogenic)
3. Brief reasoning

Protein changes to classify:
{json.dumps(protein_changes, indent=2)}

Provide a clear, concise classification for each change."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama protein classification error: {str(e)}")
        return f"Error classifying protein changes: {str(e)}"


# -----------------------------
# Literature Evidence Per Variant
# -----------------------------
def check_variant_literature_evidence(variant: Dict, disease: str) -> Dict:
    """
    Check literature evidence for each variant in the context of a specific disease.
    Uses Ollama to reason about disease relevance based on gene and protein change.
    """
    
    # Extract ACMG info safely
    acmg_data = variant.get('acmg', {})
    acmg_class = acmg_data.get('classification', 'Unknown') if isinstance(acmg_data, dict) else 'Unknown'
    acmg_evid = acmg_data.get('evidence', []) if isinstance(acmg_data, dict) else []
    
    prompt = f"""You are a genomics researcher. Evaluate the literature evidence for the following variant in the context of {disease}.

Variant Details:
- Gene: {variant.get('gene', 'Unknown')}
- Protein Change: {variant.get('AAChange', 'N/A')}
- Variant Effect: {variant.get('variant_effect', 'N/A')}
- ACMG Classification: {acmg_class}
- ACMG Evidence: {acmg_evid}

Task: 
1. Search your knowledge for known associations between this gene and {disease}
2. Assess if this specific protein change type is reported in {disease} literature
3. Determine clinical relevance (established pathogenic, likely pathogenic, unclear, benign)
4. List any known functional mechanisms or pathways
5. Indicate confidence level (high, medium, low) based on available literature
6. Do NOT hallucinate PMIDs - only mention if you're certain about evidence

Provide a structured assessment of the variant's relevance to {disease}."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        literature_assessment = response.json()["response"]
        
        return {
            "gene": variant.get("gene"),
            "AAChange": variant.get("AAChange"),
            "literature_evidence": literature_assessment,
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama literature evidence error: {str(e)}")
        return {
            "gene": variant.get("gene"),
            "AAChange": variant.get("AAChange"),
            "literature_evidence": f"Error checking literature: {str(e)}",
        }


# -----------------------------
# Variant Prioritization
# -----------------------------
def prioritize_variants(variants: List[Dict], max_variants: int = 300) -> List[Dict]:
    """
    Intelligently prioritize variants for literature mining based on:
    1. ClinVar pathogenicity score
    2. COSMIC presence
    3. Loss-of-function (LOF) impact
    4. ACMG classification
    
    Returns top N variants for analysis.
    """
    
    # Scoring function
    def variant_priority_score(v: Dict) -> float:
        score = 0.0
        
        # ACMG classification (highest priority)
        acmg = v.get('acmg', {})
        classification = acmg.get('classification', 'VUS') if isinstance(acmg, dict) else 'VUS'
        if classification == "Pathogenic":
            score += 100
        elif classification == "Likely Pathogenic":
            score += 75
        elif classification == "VUS":
            score += 25
        
        # Variant effect (loss-of-function)
        effect = str(v.get('variant_effect', '')).lower()
        if any(lof in effect for lof in ['stopgain', 'frameshift', 'splice']):
            score += 50
        elif 'missense' in effect or 'nonsynonymous' in effect:
            score += 25
        
        # Allele frequency (rare variants prioritized)
        af = v.get('af', 1.0)
        if af < 0.0001:
            score += 30
        elif af < 0.001:
            score += 20
        elif af < 0.01:
            score += 10
        
        return score
    
    # Sort by priority score
    sorted_variants = sorted(variants, key=variant_priority_score, reverse=True)
    
    # Select top variants
    prioritized = sorted_variants[:max_variants]
    
    logger.info(f"Prioritized {len(prioritized)} variants from {len(variants)} for literature analysis")
    logger.info(f"Score range: {variant_priority_score(prioritized[0]):.1f} - {variant_priority_score(prioritized[-1]):.1f}")
    
    return prioritized


# -----------------------------
# Batch Literature Analysis
# -----------------------------
def analyze_variants_literature(variants: List[Dict], disease: str, max_variants: int = 300) -> List[Dict]:
    """
    Perform literature evidence check on prioritized variants for the specified disease.
    Only analyzes top 300 variants (by default) for efficiency.
    """
    
    # Prioritize variants
    prioritized = prioritize_variants(variants, max_variants=max_variants)
    
    literature_results = []
    
    logger.info(f"Checking literature evidence for {len(prioritized)} prioritized variants in context of {disease}")
    
    for idx, variant in enumerate(prioritized):
        if idx % 50 == 0:
            logger.info(f"Processing variant {idx+1}/{len(prioritized)}")
        
        lit_evidence = check_variant_literature_evidence(variant, disease)
        literature_results.append(lit_evidence)
    
    return literature_results


# -----------------------------
# Ollama Reasoning
# -----------------------------
def ollama_reasoning(variants: List[Dict], user_prompt: str, disease: str = "") -> str:
    """
    Comprehensive variant analysis using Ollama with ACMG guidelines and actionability.
    """
    
    disease_context = f"Disease context: {disease}\n" if disease else ""
    
    prompt = f"""You are a Genetic  expert.

User question:
{user_prompt}

{disease_context}

Below are filtered variants with literature evidence assessments. Perform the following tasks:

1. Identify protein changes from AAChange and classify by functional impact
2. Assess clinical significance based on ACMG guidelines
3. Evaluate literature evidence in context of the specified disease
4. Highlight variants with strong disease association
5. Highlight actionable variants with therapeutic implications
6. Summarize key findings and clinical recommendations
7. Cite relevant literature (no hallucinated PMIDs - only mention if confident)

Return clear, evidence-based analysis.

Variants:
{json.dumps(variants, indent=2)}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama reasoning error: {str(e)}")
        return f"Error during LLM analysis: {str(e)}"


def wilms_literature_reasoning(
    wilms_variants: list,
    model: str = "llama3.1:8b",
) -> str:
    """
    LLM reasoning for Wilms tumor relevance
    """

    prompt = f"""You are a genomics researcher specializing in Wilms tumor.

Below are variants found in genes from a Wilms tumor panel with individual literature assessments.

Tasks:
1. Identify which genes are well-established in Wilms tumor pathogenesis
2. Mention known mechanisms (WT1, WTX, CTNNB1, TP53, DROSHA, etc.)
3. Flag variants likely relevant vs exploratory based on protein changes and literature evidence
4. Assess actionability for clinical management or targeted therapies
5. Highlight variants with strong Wilms tumor association
6. Cite general literature knowledge (no hallucinated PMIDs)

Return concise scientific analysis.

Variants with Literature Evidence:
{json.dumps(wilms_variants, indent=2)}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Wilms literature reasoning error: {str(e)}")
        return f"Error during Wilms analysis: {str(e)}"

# -----------------------------
# End-to-end Analysis
# -----------------------------
def analyze_variants(df: pd.DataFrame, user_prompt: str, disease: str = "Unknown", max_lit_variants: int = 300) -> Dict:
    """
    Complete variant analysis with:
    - Protein change classification
    - ACMG assessment
    - Literature evidence checking for TOP variants only
    - Disease-specific actionability evaluation
    
    Args:
        df: Filtered DataFrame with variants
        user_prompt: User's clinical question
        disease: Disease/condition for literature context
        max_lit_variants: Max variants to analyze for literature evidence
    """
    # Normalize columns first
    df = normalize_annovar_columns(df)
    
    # Use lowercase column names
    gene_col = "gene.refgene"
    exonic_col = "exonicfunc.refgene"
    aa_col = "aachange.refgene"
    
    results = []

    for _, row in df.iterrows():
        results.append({
            "gene": row.get(gene_col, "Unknown"),
            "variant_effect": row.get(exonic_col, "Unknown"),
            "af": float(row.get("gnomad_af", 0)) if pd.notna(row.get("gnomad_af")) else 0,
            "depth": int(row.get("depth", 0)) if pd.notna(row.get("depth")) else 0,
            "quality": float(row.get("qual", 0)) if pd.notna(row.get("qual")) else 0,
            "hgvsc": row.get("hgvsc", ""),
            "hgvsp": row.get("hgvsp", ""),
            "AAChange": row.get(aa_col, ""),
            "chrom": str(row.get("chrom", "")),
            "pos": str(row.get("pos", "")),
            "ref": str(row.get("ref", "")),
            "alt": str(row.get("alt", "")),
            "acmg": acmg_classification(row),
            "actionability": cancer_actionability(row.get(gene_col, "Unknown")),
        })

    logger.info(f"Created {len(results)} variant records for analysis")

    # Classify protein changes (all variants)
    protein_classification = classify_protein_changes(results)
    
    # Check literature evidence for TOP variants only
    logger.info(f"Prioritizing top {max_lit_variants} variants for literature analysis")
    literature_evidence = analyze_variants_literature(results, disease, max_variants=max_lit_variants)
    
    # Create mapping of prioritized variants for faster lookup
    lit_genes_set = {(lit['gene'], lit['AAChange']) for lit in literature_evidence}
    
    # Add literature evidence back to results (only for analyzed variants)
    for variant in results:
        if (variant['gene'], variant['AAChange']) in lit_genes_set:
            for lit_result in literature_evidence:
                if lit_result['gene'] == variant['gene'] and lit_result['AAChange'] == variant['AAChange']:
                    variant["literature_evidence"] = lit_result.get("literature_evidence", "")
                    break
        else:
            variant["literature_evidence"] = "Not prioritized for literature analysis"
    
    # Comprehensive reasoning on prioritized variants
    prioritized_for_summary = prioritize_variants(results, max_variants=max_lit_variants)
    summary = ollama_reasoning(prioritized_for_summary, user_prompt, disease)

    return {
        "variant_count": len(results),
        "prioritized_count": len(literature_evidence),
        "disease": disease,
        "variants": results,
        "prioritized_variants": prioritized_for_summary,
        "protein_classification": protein_classification,
        "literature_evidence": literature_evidence,
        "summary": summary,
    }