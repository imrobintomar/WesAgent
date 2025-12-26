# backend/export_utils.py

from pathlib import Path
import pandas as pd


def export_full_vcf_csv(
    df: pd.DataFrame,
    output_dir: str = "exports",
    filename: str = "wes_full_variants_with_burden.csv",
) -> str:
    """
    Exports FULL variant-level dataset with burden annotations.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    df.to_csv(path, index=False)

    return str(path)
