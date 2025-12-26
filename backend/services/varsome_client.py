import os
import requests

VARSOME_TOKEN = os.getenv("VARSOME_API_TOKEN")


def classify_variant(chrom, pos, ref, alt, genome="hg38"):
    """
    VarSome ACMG classification (PRIMARY)
    """
    chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    key = f"{chrom}:{pos}:{ref}:{alt}"

    url = f"https://api.varsome.com/lookup/{key}/{genome}"
    headers = {"Authorization": f"Token {VARSOME_TOKEN}"}
    params = {"add-ACMG-annotation": 1}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return {
            "ACMG": "NA",
            "ACMG_SOURCE": "VarSome",
            "ACMG_RULES": [],
            "TRANSCRIPT": None,
        }

    data = r.json()
    acmg = data.get("acmg_annotation", {})

    return {
        "ACMG": acmg.get("verdict", {}).get("verdict", "NA"),
        "ACMG_SOURCE": "VarSome",
        "ACMG_RULES": acmg.get("classifications", []),
        "TRANSCRIPT": acmg.get("transcript"),
    }
