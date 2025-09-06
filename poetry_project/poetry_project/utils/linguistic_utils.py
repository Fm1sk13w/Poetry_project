"""
Linguistic analysis utilities for poetry_project.

Provides functions for syllable counting, adjective frequency analysis,
and other text-based linguistic metrics used in poem and author analysis.
"""

import pronouncing
import re
import spacy
from typing import Tuple, Any

_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def adjectives_plus_adverbs_ratio(poem: Any) -> Tuple[int, float]:
    """
    Count adjectives + adverns in a poem and compute ratio to total words.

    Args:
        poem: Object with a `content` attribute containing the poem text.

    Returns:
        Tuple of (adjective_adverb_count, adjective_adverb_ratio).
        Ratio is 0.0 if poem has no words.
    """
    doc = _nlp(poem.content)
    adj_adv_count = sum(1 for token in doc if token.pos_ in ["ADJ", "ADV"])
    total_words = len([t for t in doc if t.is_alpha])
    ratio = adj_adv_count / total_words if total_words > 0 else 0.0
    return adj_adv_count, ratio


def count_syllables_in_line(line: str) -> int:
    """
    Count syllables in a single line of text.

    Tries CMU dictionary first; falls back to vowel clusters.

    Args:
        line: A line of text.

    Returns:
        Estimated number of syllables in the line.
    """
    words = re.findall(r"\b\w+\b", line.lower())
    total = 0
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            total += pronouncing.syllable_count(phones[0])
        else:
            total += len(re.findall(r"[aeiouy]+", word))
    return total