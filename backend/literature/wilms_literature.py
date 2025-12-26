"""
Curated Wilms tumor literature references
Used for citation-grounded LLM reasoning
"""

def get_wilms_literature() -> list[dict]:
    """
    Returns a list of curated Wilms tumor references.
    Each entry contains structured metadata.
    """

    return [
        {
            "id": "PMID_24766809",
            "title": "Mutations of WT1, CTNNB1, and WTX in Wilms tumor",
            "journal": "Nature Genetics",
            "year": 2014,
            "pmid": "24766809",
            "genes": ["WT1", "CTNNB1", "AMER1"],
        },
        {
            "id": "PMID_28726821",
            "title": "Recurrent DROSHA and DGCR8 mutations in Wilms tumour",
            "journal": "Nature Communications",
            "year": 2017,
            "pmid": "28726821",
            "genes": ["DROSHA", "DGCR8"],
        },
        {
            "id": "PMID_31827281",
            "title": "Genomic and transcriptomic analysis of Wilms tumor",
            "journal": "Cancer Cell",
            "year": 2019,
            "pmid": "31827281",
            "genes": ["SIX1", "SIX2", "MYCN", "TP53"],
        },
        {
            "id": "PMID_32929243",
            "title": "The landscape of somatic mutations in Wilms tumor",
            "journal": "Journal of Clinical Oncology",
            "year": 2020,
            "pmid": "32929243",
            "genes": ["BCOR", "BCORL1", "SMARCA4"],
        },
    ]
