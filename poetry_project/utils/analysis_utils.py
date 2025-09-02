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
      - nationality (str)
      - avg_poem_length (words)
      - rhyme_percentage
      - poem_count (number of poems scraped)
      - constant_syllable_percentage (percentage of poems where all lines have the same syllable count)

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
        total_poems = len(author.poems)

        if total_poems == 0:
            rhyme_pct = 0.0
            constant_syllable_pct = 0.0
        else:
            rhymed = sum(
                poem_has_rhymes(p, tolerance=rhyme_tolerance)
                for p in author.poems
            )
            rhyme_pct = (rhymed / total_poems) * 100

            constant_syllable_count = sum(
                1
                for p in author.poems
                if p.number_of_syllables_in_lines
                and len(set(p.number_of_syllables_in_lines)) == 1
            )
            constant_syllable_pct = (constant_syllable_count / total_poems) * 100

        records.append({
            "author": author.name,
            "birth_year": birth_year,
            "nationality": author.nationality,
            "avg_poem_length": avg_length,
            "rhyme_percentage": rhyme_pct,
            "poem_count": total_poems,
            "constant_syllable_percentage": constant_syllable_pct
        })

    return pd.DataFrame.from_records(records)