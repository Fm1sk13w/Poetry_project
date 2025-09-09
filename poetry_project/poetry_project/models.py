"""
Data models for Poem and Author, with relationships and derived fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from poetry_project.utils.linguistic_utils import count_syllables_in_line


@dataclass
class Author:
    """
    Represents a poet with metadata and a collection of poems.
    """
    def __init__(self, name, birth_year=None, nationality=None):
            self.name = name
            self.birth_year = birth_year
            self.nationality = nationality
            self.poems = []
            
    def get_poem_by_title(self, title: str):
        """
        Return the Poem object for this author matching the given title.

        Args:
            title: Poem title to search for (case-insensitive).

        Returns:
            Poem instance if found, else None.
        """
        title_lower = title.strip().lower()
        return next(
            (poem for poem in self.poems if poem.title.strip().lower() == title_lower),
            None
        )
    
    def average_poem_length(self) -> float:
        """
        Calculate the average number of words per poem.

        Returns:
            The average poem length in words, or 0.0 if no poems are present.
        """
        if not self.poems:
            return 0.0
        total_words = sum(p.number_of_words for p in self.poems)
        return total_words / len(self.poems)
    
    def __str__(self):
        """Return a human-readable summary of the author."""
        return (
            f"{self.name} — "
            f"Birth year: {self.birth_year or 'Unknown'} | "
            f"Nationality: {self.nationality or 'Unknown'} | "
            f"Poems in database: {len(self.poems)}"
        )


@dataclass
class Poem:
    """
    Represents a single poem and computes basic metrics on creation.
    """
    title: str
    author: Author
    content: str
    source_url: str
    number_of_words: int = field(init=False)
    number_of_lines: int = field(init=False)
    number_of_syllables_in_lines: List[int] = field(init=False)

    def __post_init__(self) -> None:
        """
        Derive word, line, and syllable counts after initialization.
        """
        self.number_of_words = len(self.content.split())
        self.number_of_lines = len(self.content.splitlines())
        self.number_of_syllables_in_lines = [
            count_syllables_in_line(line)
            for line in self.content.splitlines()
            if line.strip()
        ]

    def __str__(self) -> str:
        """
        A human-readable representation.
        """
        return f'"{self.title}" by {self.author.name}'
