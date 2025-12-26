import pandas as pd


def annotate_wilms_burden(df: pd.DataFrame, wilms_genes: set) -> pd.DataFrame:
    """
    Adds Wilms tumor–specific annotations
    """
    df = df.copy()
    
    # Try multiple common gene column names
    gene_col = None
    for col in ["gene.refgene", "gene", "gene_name", "symbol"]:
        if col in df.columns:
            gene_col = col
            break
            
    if gene_col is None:
        # If no common name, look for something containing 'gene'
        for col in df.columns:
            if 'gene' in col.lower():
                gene_col = col
                break

    if gene_col:
        df["GENE_SYMBOL"] = (
            df[gene_col]
            .astype(str)
            .str.split(";")
            .str[0]
            .str.upper()
        )
    else:
        df["GENE_SYMBOL"] = "UNKNOWN"

    df["IS_WILMS_GENE"] = df["GENE_SYMBOL"].isin(wilms_genes)

    return df


def wilms_gene_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gene-level mutation burden for Wilms panel
    """
    return (
        df[df["IS_WILMS_GENE"]]
        .groupby("GENE_SYMBOL")
        .size()
        .reset_index(name="variant_count")
        .sort_values("variant_count", ascending=False)
    )
