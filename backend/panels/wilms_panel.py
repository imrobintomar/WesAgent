"""
Wilms Tumor Gene Panel
Embedded, file-independent, research-safe

Source examples:
- COSMIC
- ClinGen
- OMIM
- Recent Wilms tumor genomics literature
"""

def load_wilms_panel() -> set:
    """
    Returns a set of Wilms tumor–associated gene symbols.
    All symbols are normalized to uppercase.

    No external files required.
    """

    wilms_genes = {
        # Core Wilms tumor drivers
        "WT1",
        "CTNNB1",
        "TP53",
        "DROSHA",
        "DGCR8",
        "DIS3L2",
        "SIX1",
        "SIX2",
        "MYCN",
        "AMER1",   # WT-associated tumor suppressor
        "REST",
        "TRIM28",

        # Chromatin / epigenetic regulators
        "BCOR",
        "BCORL1",
        "KDM6A",
        "KMT2D",
        "ARID1A",
        "ARID1B",
        "SMARCA4",
        "SMARCB1",
        "EZH2",

        # miRNA / RNA processing pathway
        "XPO5",
        "TARBP2",
        "LIN28A",
        "LIN28B",

        # Growth / signaling
        "IGF2",
        "H19",
        "CDKN2A",
        "CDKN1C",
        "PIK3CA",
        "PTEN",

        # DNA damage / genome stability
        "BRCA2",
        "BRCA1",
        "ATM",
        "ATR",
        "CHEK2",

        # Reported recurrent or rare associations
        "NF1",
        "KRAS",
        "NRAS",
        "HRAS",
        "FGFR2",
        "FGFR3",
        "ALK",
        "BRAF",
    }

    # Normalize (defensive, future-proof)
    return {g.upper().strip() for g in wilms_genes}
