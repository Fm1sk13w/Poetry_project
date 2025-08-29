"""
Helper functions to compile author-level metrics into a DataFrame.
"""

from typing import List, Dict, Any
import pandas as pd

from poetry_project.models import Author
from poetry_project.utils.rhyme_utils import poem_has_rhymes


def compute_author_metrics(
    authors: List[Author],
    rhyme_tolerance: float = 0.7
) -> pd.DataFrame:
    """
    Build a DataFrame of author metrics:
      - birth_year (int)
      - avg_poem_length (words)
      - rhyme_percentage

    Args:
        authors: List of Author instances.
        rhyme_tolerance: Threshold for poem_has_rhymes.

    Returns:
        A pandas DataFrame with one row per author.
    """
    records: List[Dict[str, Any]] = []

    for author in authors:
        # Skip if birth_year is missing or non-numeric
        if author.birth_year is None or not author.birth_year.isdigit():
            continue

        birth_year = int(author.birth_year)
        avg_length = author.average_poem_length()
        total = len(author.poems)
        rhymed = sum(poem_has_rhymes(p, tolerance=rhyme_tolerance) for p in author.poems)
        rhyme_pct = (rhymed / total * 100) if total else 0.0

        records.append({
            "author": author.name,
            "birth_year": birth_year,
            "avg_poem_length": avg_length,
            "rhyme_percentage": rhyme_pct
        })

    df = pd.DataFrame.from_records(records)
    return df