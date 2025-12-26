import os
import requests

FRANKLIN_TOKEN = os.getenv("FRANKLIN_API_TOKEN")

FRANKLIN_PARSE_URL = "https://franklin.genoox.com/api/parse_search"
FRANKLIN_SEARCH_URL = "https://api.genoox.com/v2/search/snp/"


def parse_variant(search_text: str):
    """
    Parse ANY variant string:
    - VHL:c.293A>G
    - NM_000551.4:c.293A>G
    - rs397515359
    - chr3:10183824 A-G
    """
    headers = {
        "Authorization": f"Bearer {FRANKLIN_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"search_text_input": search_text}

    r = requests.post(FRANKLIN_PARSE_URL, headers=headers, json=payload, timeout=20)
    if r.status_code != 200:
        return None

    data = r.json()
    variant = data.get("best_variant_option") or (
        data.get("snp_variants") or [None]
    )[0]

    if not variant:
        return None

    return {
        "chrom": variant.get("chrom"),
        "pos": variant.get("pos"),
        "ref": variant.get("ref"),
        "alt": variant.get("alt"),
        "transcript": variant.get("canonical_tanscript"),
    }


def classify_variant(chrom, pos, ref, alt):
    """
    Franklin SNP classification using normalized chr-pos-ref-alt
    """
    search_text = f"{chrom}-{pos}-{ref}-{alt}"
    headers = {
        "Authorization": f"Bearer {FRANKLIN_TOKEN}",
        "Content-Type": "application/json",
    }

    r = requests.get(
        FRANKLIN_SEARCH_URL,
        headers=headers,
        params={"search_text": search_text},
        timeout=20,
    )

    if r.status_code != 200:
        return {
            "FRANKLIN_ACMG": "NA",
            "FRANKLIN_RULES": [],
            "FRANKLIN_LINK": None,
        }

    data = r.json()
    cls = data.get("classification", {})

    return {
        "FRANKLIN_ACMG": cls.get("acmg_classification", "NA"),
        "FRANKLIN_RULES": cls.get("acmg_rules", []),
        "FRANKLIN_LINK": data.get("variant_franklin_link"),
    }
