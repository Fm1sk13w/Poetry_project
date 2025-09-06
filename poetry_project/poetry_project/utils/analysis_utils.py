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
from poetry_project.utils.rhyme_utils import poem_has_rhymes, rhyme_check
from poetry_project.utils.linguistic_utils import adjectives_plus_adverbs_ratio


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

import re
from typing import Set

def rhyme_density_local_all_words(poem, window: int = 4) -> float:
    """
    Calculate rhyme density for all words in a poem, checking only the next `window` lines.

    Args:
        poem: Poem object with a `.content` attribute (string).
        window: Number of subsequent lines to check for rhymes.

    Returns:
        Ratio of rhyming words to total words in the poem.
    """
    # Split into lines and extract words
    lines = [l.strip() for l in poem.content.split("\n") if l.strip()]
    tokenized_lines = [
        re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", line.lower()) for line in lines
    ]

    total_words = sum(len(words) for words in tokenized_lines)
    if total_words == 0:
        return 0.0

    rhyming_words: Set[tuple[int, str]] = set()  # (line_index, word)

    for i, words in enumerate(tokenized_lines):
        for w in words:
            # Compare with words in the next `window` lines
            for j in range(i + 1, min(i + 1 + window, len(tokenized_lines))):
                for w2 in tokenized_lines[j]:
                    if w2 == w:
                        continue  # skip identical words
                    if rhyme_check(w, w2):
                        rhyming_words.add((i, w))
                        rhyming_words.add((j, w2))

    return len(rhyming_words) / total_words



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
        - avg_adjective_adverb_ratio (fraction of (adjectives+adverbs) in poems)

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
            avg_adj_adv_ratio = 0.0
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
                rhyme_density_local_all_words(p) for p in author.poems
            ]))

            avg_adj_adv_ratio = float(np.mean([
            adjectives_plus_adverbs_ratio(p)[1] for p in author.poems
            ]))

        records.append({
            "author": author.name,
            "birth_year": birth_year,
            "nationality": author.nationality,
            "avg_poem_length": avg_length,
            "rhyme_percentage": rhyme_pct,
            "poem_count": total_poems,
            "constant_syllable_percentage": constant_syllable_pct,
            "avg_rhyme_density_all_words": avg_rhyme_density,
            "avg_adjective_adverb_ratio": avg_adj_adv_ratio
        })

    return pd.DataFrame.from_records(records)
