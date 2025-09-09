"""
Utility functions for compiling and analyzing author-level poetry metrics.

This module aggregates poem-level statistics into author-level summaries
and provides additional analytical functions for deeper statistical
and linguistic exploration.

"""

from __future__ import annotations

import pandas as pd
import numpy as np
import re
import pronouncing
import spacy
from functools import lru_cache
from typing import List, Dict, Any, Set
from collections import Counter


from poetry_project.models import Author, Poem
from poetry_project.utils.rhyme_utils import poem_has_rhymes, rhyme_check
from poetry_project.utils.linguistic_utils import adjectives_plus_adverbs_ratio
from poetry_project.utils.lexical_diversity import compute_author_lexical_breadth

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


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_author_metrics(
    authors: List["Author"],
    rhyme_tolerance: float = 0.7
) -> pd.DataFrame:
    """
    Build a DataFrame of author metrics.

    Metrics include:
        - birth_year (int)
        - nationality (str)
        - avg_poem_length (words)
        - end_line_rhyme_poems_pct (end-line rhyming poems)
        - poem_count
        - constant_syllable_pct
        - avg_rhyme_density (fraction of rhyming words in poems)
        - avg_adj_adv_ratio (fraction of (adjectives+adverbs) in poems)
        - lexical_mtld (Measure of Textual Lexical Diversity)

    Args:
        authors: List of Author instances.
        rhyme_tolerance: Threshold for poem_has_rhymes.

    Returns:
        A pandas DataFrame with one row per author.
    """
    records: List[Dict[str, Any]] = []

    # Precompute lexical diversity scores for all authors
    lexical_mtld_scores = compute_author_lexical_breadth(
        authors,
        nlp,
        method="mtld",
        preprocess_kwargs=dict(remove_stopwords=True, remove_propn=True)
    )

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
            "end_line_rhyme_poems_pct": rhyme_pct,
            "poem_count": total_poems,
            "constant_syllable_pct": constant_syllable_pct,
            "avg_rhyme_density_all_words": avg_rhyme_density,
            "avg_adj_adv_ratio": avg_adj_adv_ratio,
            "lexical_mtld": lexical_mtld_scores.get(author.name, float("nan"))
        })

    return pd.DataFrame.from_records(records)


def normalize_nationality(nationality: str) -> str:
    """
    Normalize nationality names to a consistent set of country labels.

    This function standardizes variations in nationality naming so that
    related regions are grouped under a single label. Constituent countries 
    of the United Kingdom such as "England", "Wales","Scotland", 
    and "Northern Ireland" are all mapped to "United Kingdom".
    This ensures consistency in downstream analyses and visualizations.

    Args:
        nationality (str): The original nationality string to normalize.

    Returns:
        str: The normalized nationality label.

    Examples:
        >>> normalize_nationality("England")
        'United Kingdom'
    """

    uk_variants = {"England", "Wales", "Scotland", "Northern Ireland"}
    if nationality in uk_variants:
        return "United Kingdom"
    return nationality


def compute_country_stats(authors: List[Author], metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute country-level statistics from authors and metrics.

    Returns:
        DataFrame with one row per country, including:
        - Poets: number of poets from that country
        - Total Poems: total number of poems
        - Authors Avg Birth Year: average birth year of poets
        - Avg Poem Length
        - Avg Adjective + Adverb Ratio
        - Avg Rhyme Density
        - Constant Syllable %
    """
    # Normalize nationality labels if needed
    metrics_df["nationality"] = metrics_df["nationality"].apply(normalize_nationality)

    # Count poets per country
    poet_counts = metrics_df["nationality"].value_counts()

    # Total poems per country
    total_poems = (
        metrics_df.groupby("nationality")["poem_count"]
        .sum()
        .rename("Total Poems")
    )

    # Average birth year
    avg_birth_year = (
        metrics_df.groupby("nationality")["birth_year"]
        .mean()
        .round(0)
        .astype(int)
        .rename("Authors Avg Birth Year")
    )

    # Other averaged metrics
    averaged_metrics = (
        metrics_df.groupby("nationality")
        .agg({
            "avg_poem_length": "mean",
            "avg_adj_adv_ratio": "mean",
            "avg_rhyme_density_all_words": "mean",
            "constant_syllable_pct": "mean"
        })
        .rename(columns={
            "avg_poem_length": "Avg Poem Length",
            "avg_adj_adv_ratio": "Avg Adjective + Adverb Ratio",
            "avg_rhyme_density_all_words": "Avg Rhyme Density",
            "constant_syllable_pct": "Constant Syllable %"
        })
        .round(2)
    )

    # Combine all stats
    country_stats = pd.DataFrame({
        "Poets": poet_counts
    }).join([
        total_poems,
        avg_birth_year,
        averaged_metrics
    ])

    return country_stats.sort_values("Poets", ascending=False)


# List of known unwanted fragments
UNWANTED_FRAGMENTS = {"ing", "ly", "er", "ed", "es", "est"}

def most_common_words(authors, top_n: int = 50, remove_stopwords: bool = True, min_len: int = 3) -> pd.DataFrame:
    """
    Find the most common words across all poems in the database, efficiently.

    Args:
        authors: List of Author objects with .poems and .content.
        top_n: Number of top words to return.
        remove_stopwords: Whether to exclude stopwords.
        min_len: Minimum length of token to include.

    Returns:
        DataFrame with columns ['word', 'count'] sorted by count descending.
    """
    # Gather all poem texts
    texts = [p.content for a in authors for p in a.poems if getattr(p, "content", None)]

    counter = Counter()

    # Process in batches for speed
    for doc in nlp.pipe(texts, batch_size=50):
        words = [
            t.lemma_.lower()
            for t in doc
            if not t.is_punct
            and not t.is_space
            and not t.like_num
            and (not remove_stopwords or not t.is_stop)
            and len(t.lemma_) >= min_len
            and t.lemma_.isalpha()
            and t.lemma_.lower() not in UNWANTED_FRAGMENTS
        ]
        counter.update(words)

    return pd.DataFrame(counter.most_common(top_n), columns=["word", "count"])


def detect_trendy_words(authors, n_bins: int = 4, min_count: int = 5, remove_stopwords: bool = True, min_len: int = 3):
    """
    Split poems into n_bins by author's birth_year and find words with uneven usage.

    Args:
        authors: List of Author objects with .birth_year and .poems.
        n_bins: Number of time bins to split by birth_year.
        min_count: Minimum total occurrences of a word to be considered.
        remove_stopwords: Whether to exclude stopwords.
        min_len: Minimum token length to include.

    Returns:
        freq_df: DataFrame with relative frequencies per bin for each word.
        bin_edges: List of bin edges used for splitting.
    """
    # Gather (word, bin) pairs
    rows = []
    years = [int(a.birth_year) for a in authors if a.birth_year and str(a.birth_year).isdigit()]
    if not years:
        raise ValueError("No valid birth_year data to bin by time.")

    min_year, max_year = min(years), max(years)
    bin_edges = np.linspace(min_year, max_year, n_bins + 1)

    texts_with_bins = []
    for author in authors:
        if not author.birth_year or not str(author.birth_year).isdigit():
            continue
        year = int(author.birth_year)
        bin_idx = np.digitize(year, bin_edges) - 1
        for poem in author.poems:
            if getattr(poem, "content", None):
                texts_with_bins.append((poem.content, bin_idx))

    # Process in batches
    for text, bin_idx in texts_with_bins:
        doc = nlp(text)
        for t in doc:
            if t.is_punct or t.is_space or t.like_num:
                continue
            if remove_stopwords and t.is_stop:
                continue
            lemma = t.lemma_.lower()
            if len(lemma) < min_len or not lemma.isalpha():
                continue
            if lemma in UNWANTED_FRAGMENTS:
                continue
            rows.append((lemma, bin_idx))

    df = pd.DataFrame(rows, columns=["word", "bin"])
    total_per_bin = df.groupby("bin").size()

    # Relative frequency per bin
    freq = df.groupby(["word", "bin"]).size().unstack(fill_value=0)
    freq = freq.div(total_per_bin, axis=1)

    # Filter by min_count
    total_counts = df.groupby("word").size()
    freq = freq.loc[total_counts[total_counts >= min_count].index]

    # Detect uneven usage: std deviation across bins
    freq["std_dev"] = freq.std(axis=1)
    freq = freq.sort_values("std_dev", ascending=False)

    return freq, bin_edges