# backend/stats_engine.py

import pandas as pd
from typing import Dict, Tuple


# -------------------------------------------------
# Variant Type Annotation
# -------------------------------------------------
def annotate_variant_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify variants into SNV or INDEL based on ExonicFunc.refGene
    """

    def classify(exonic_func: str) -> str:
        if pd.isna(exonic_func):
            return "Unknown"
        exonic_func = exonic_func.lower()
        if "snv" in exonic_func:
            return "SNV"
        if "frameshift" in exonic_func or "insertion" in exonic_func or "deletion" in exonic_func:
            return "INDEL"
        return "Other"

    df = df.copy()
    df["variant_type"] = df["ExonicFunc.refGene"].apply(classify)
    return df


# -------------------------------------------------
# Gene-level Mutation Burden
# -------------------------------------------------
def compute_gene_mutation_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mutation burden per gene
    """

    gene_burden = (
        df.groupby("Gene.refGene")
        .size()
        .reset_index(name="variant_count")
        .sort_values("variant_count", ascending=False)
    )

    return gene_burden


# -------------------------------------------------
# SNV / INDEL Burden
# -------------------------------------------------
def compute_variant_type_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns SNV vs INDEL burden
    """

    if "variant_type" not in df.columns:
        df = annotate_variant_type(df)

    variant_burden = (
        df["variant_type"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "variant_type", "variant_type": "count"})
    )

    return variant_burden


# -------------------------------------------------
# Functional Impact Burden
# -------------------------------------------------
def compute_functional_impact_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts functional classes (missense, frameshift, stopgain, etc.)
    """

    impact_burden = (
        df["ExonicFunc.refGene"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "functional_class", "ExonicFunc.refGene": "count"})
    )

    return impact_burden


# -------------------------------------------------
# Combined Summary (for LLM or reports)
# -------------------------------------------------
def compute_burden_summary(df: pd.DataFrame) -> Dict:
    """
    High-level summary statistics suitable for LLM reasoning
    """

    df = annotate_variant_type(df)

    summary = {
        "total_variants": int(len(df)),
        "unique_genes": int(df["Gene.refGene"].nunique()),
        "snv_count": int((df["variant_type"] == "SNV").sum()),
        "indel_count": int((df["variant_type"] == "INDEL").sum()),
        "top_mutated_genes": (
            df.groupby("Gene.refGene")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ),
        "functional_distribution": (
            df["ExonicFunc.refGene"]
            .value_counts()
            .head(10)
            .to_dict()
        ),
    }

    return summary
