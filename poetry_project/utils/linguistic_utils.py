import pronouncing
import re

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