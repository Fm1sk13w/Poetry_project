"""
wikidata_person_info.py

Provides a function to query Wikidata for a person's birth year and country of citizenship,
with normalization of historical and formal country names into common English labels.

Author: PoetryProjectBot
Contact: your_email@example.com
"""

from functools import lru_cache
from typing import Optional, Tuple
import requests
import re

# Hand-curated overrides for English-speaking countries (historical & formal)
COMMON_NAME_OVERRIDES: dict[str, str] = {
    # United Kingdom & predecessors
    "Kingdom of England": "England",
    "Kingdom of Scotland": "Scotland",
    "Kingdom of Ireland": "Ireland",
    "Kingdom of Great Britain": "United Kingdom",
    "United Kingdom of Great Britain and Ireland": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",

    # Ireland
    "Irish Free State": "Ireland",
    "Republic of Ireland": "Ireland",

    # United States
    "United States of America": "United States",
    "USA": "United States",
    "Thirteen Colonies": "United States",
    "British America": "United States",
    "Confederate States of America": "United States",

    # Canada
    "Dominion of Canada": "Canada",
    "Province of Canada": "Canada",
    "British North America": "Canada",
    "Canadian Confederation": "Canada",

    # Australia
    "Commonwealth of Australia": "Australia",
    "Federation of Australia": "Australia",
    "Colony of New South Wales": "Australia",
    "Colony of Victoria": "Australia",
    "Colony of Tasmania": "Australia",

    # New Zealand
    "Dominion of New Zealand": "New Zealand",
}


def normalize_country(label: str) -> str:
    """
    Normalize Wikidata country labels into common English names.

    Args:
        label: Raw country label from Wikidata.

    Returns:
        A simplified, standardized country name.
    """
    # 1. Check explicit overrides
    if label in COMMON_NAME_OVERRIDES:
        return COMMON_NAME_OVERRIDES[label]

    # 2. Strip common prefixes
    stripped = re.sub(
        r'^(Kingdom|Republic|State|Federation|Commonwealth|Dominion|Colony|Principality)\s+of\s+',
        '',
        label,
        flags=re.IGNORECASE
    ).strip()

    return stripped


@lru_cache(maxsize=8)
def get_birth_and_nationality_wikidata(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Query Wikidata for a person's birth year and country of citizenship.

    Args:
        name: English label of the person (e.g., "William Shakespeare").

    Returns:
        A tuple (birth_year, normalized_country), or (None, None) if not found or failed.
    """
    query = f"""
    SELECT ?dob ?countryLabel WHERE {{
      ?person rdfs:label "{name}"@en.
      OPTIONAL {{ ?person wdt:P569 ?dob. }}
      OPTIONAL {{ ?person wdt:P27 ?country. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "PoetryProjectBot/1.0"
    }

    try:
        response = requests.get(url, params={"query": query}, headers=headers, timeout=10)
        response.raise_for_status()
        bindings = response.json().get("results", {}).get("bindings", [])
        if not bindings:
            return None, None

        row = bindings[0]
        dob_iso = row.get("dob", {}).get("value")
        raw_country = row.get("countryLabel", {}).get("value")

        birth_year = dob_iso[:4] if dob_iso else None
        country = normalize_country(raw_country) if raw_country else None

        return birth_year, country

    except requests.exceptions.RequestException as error:
        print(f"Wikidata query failed for '{name}': {error}")
        return None, None
