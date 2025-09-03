"""
Utility functions for compiling and analyzing author-level poetry metrics.

This module aggregates poem-level statistics into author-level summaries
and provides additional analytical functions for deeper statistical
and linguistic exploration.

"""

from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re
from functools import lru_cache
import pronouncing

from poetry_project.models import Author, Poem
from poetry_project.utils.rhyme_utils import poem_has_rhymes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _phones(word: str) -> list[str]:
    """Cached CMU dictionary lookup for a word."""
    return pronouncing.phones_for_word(word)

@lru_cache(maxsize=None)
def _rhyming_part(word: str) -> str | None:
    """Cached rhyming part lookup for a word."""
    phones = _phones(word)
    return pronouncing.rhyming_part(phones[0]) if phones else None

def _rhyme_density_all_words(poem: Poem) -> float:
    """
    Calculate rhyme density across all words in a poem.

    Rhyme density = fraction of words that share a rhyming part
    with at least one other word in the same poem.
    """
    words = re.findall(r"\b\w+\b", poem.content.lower())
    if len(words) < 2:
        return 0.0

    rhyme_parts = [_rhyming_part(w) for w in words]
    rhyming_groups: Dict[str, int] = {}
    for rp in rhyme_parts:
        if rp:
            rhyming_groups[rp] = rhyming_groups.get(rp, 0) + 1

    rhymed_word_count = sum(
        count for count in rhyming_groups.values() if count > 1
    )

    return rhymed_word_count / len(words)


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_author_metrics(
    authors: List[Author],
    rhyme_tolerance: float = 0.7
) -> pd.DataFrame:
    """
    Build a DataFrame of author metrics.

    Metrics include:
        - birth_year (int)
        - nationality (str)
        - avg_poem_length (words)
        - rhyme_percentage (end-line rhymes)
        - poem_count
        - constant_syllable_percentage
        - avg_rhyme_density_all_words (fraction of rhyming words in poems)

    Args:
        authors: List of Author instances.
        rhyme_tolerance: Threshold for poem_has_rhymes.

    Returns:
        A pandas DataFrame with one row per author.
    """
    records: List[Dict[str, Any]] = []

    for author in authors:
        if author.birth_year is None or not author.birth_year.isdigit():
            continue

        birth_year = int(author.birth_year)
        avg_length = author.average_poem_length()
        total_poems = len(author.poems)

        if total_poems == 0:
            rhyme_pct = 0.0
            constant_syllable_pct = 0.0
            avg_rhyme_density = 0.0
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

            avg_rhyme_density = float(np.mean([
                _rhyme_density_all_words(p) for p in author.poems
            ]))

        records.append({
            "author": author.name,
            "birth_year": birth_year,
            "nationality": author.nationality,
            "avg_poem_length": avg_length,
            "rhyme_percentage": rhyme_pct,
            "poem_count": total_poems,
            "constant_syllable_percentage": constant_syllable_pct,
            "avg_rhyme_density_all_words": avg_rhyme_density
        })

    return pd.DataFrame.from_records(records)
