"""
Utility functions for rhyme checking and syllable counting.
"""

import pronouncing
import re
from typing import Set
from poetry_project.models import Poem


def rhyme_check(word_1: str, word_2: str) -> bool:
    """
    Determine whether two words rhyme, using the CMU Pronouncing Dictionary.

    Args:
        word_1: First word to compare.
        word_2: Second word to compare.

    Returns:
        True if any pronunciation of word_1 rhymes with any pronunciation of word_2.
    """
    phones1 = pronouncing.phones_for_word(word_1)
    phones2 = pronouncing.phones_for_word(word_2)
    return any(
        pronouncing.rhyming_part(p1) == pronouncing.rhyming_part(p2)
        for p1 in phones1
        for p2 in phones2
    )


def get_last_word(line: str) -> str:
    """
    Extract the last word from a line of text.
    """
    words = re.findall(r"\b\w+\b", line.lower())
    return words[-1] if words else ""

def poem_has_rhymes(poem: Poem, tolerance: float = 0.7) -> bool:
    """
    Determine whether a poem uses end-line rhymes (vs. blank verse).

    Args:
        poem: Poem instance whose content will be analyzed.
        tolerance: Fraction of lines that must rhyme with at least one other
                   line for us to call the poem “rhymed” (0 < tolerance <= 1).

    Returns:
        True if proportion of rhyming lines >= tolerance, False otherwise.
    """
    lines = [line.strip() for line in poem.content.splitlines() if line.strip()]
    total = len(lines)
    if total < 2:
        return False

    last_words = [get_last_word(line) for line in lines]
    rhymed: Set[int] = set()

    for i, word_i in enumerate(last_words):
        for j, word_j in enumerate(last_words[i + 1:], start=i + 1):
            if word_i and word_j and rhyme_check(word_i, word_j):
                rhymed.add(i)
                rhymed.add(j)

    rhyme_rate = len(rhymed) / total
    return rhyme_rate >= tolerance