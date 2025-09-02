"""
High-level pipeline to assemble a complete poem database with metadata.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from poetry_project.models import Author, Poem
from poetry_project.scraper.author_scraper import (
    get_poets_by_poem_range,
    get_poem_urls_by_author
)
from poetry_project.scraper.poem_scraper import scrape_poem_content
from poetry_project.metadata.wikidata import get_birth_and_nationality_wikidata
from poetry_project.utils.persistence import (
    load_author_checkpoints,
    save_author_checkpoint
)

def get_complete_poems_by_author(
    author_name: str,
    max_workers: int = 10
) -> Author:
    """
    Build an Author object populated with all poems and metadata.

    Args:
        author_name: Full name of the poet.
        max_workers: Number of threads for parallel poem scraping.

    Returns:
        An Author instance with birth_year, nationality, and poems list.
    """
    birth_year, nationality = get_birth_and_nationality_wikidata(author_name)
    author = Author(name=author_name, birth_year=birth_year,
                    nationality=nationality)

    urls = get_poem_urls_by_author(author_name)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for poem in pool.map(lambda u: scrape_poem_content(u, author), urls):
            pass  # scrape_poem_content already appends to author.poems

    return author


def build_poems_database(
    poets: List[Tuple[str, int]],
    max_workers: int = 4
) -> List[Poem]:
    """
    Flatten complete poem lists across multiple authors.

    Args:
        poets: List of (author_name, poem_count).
        max_workers: Threads to parallelize authors.

    Returns:
        A flat list of all Poem instances.
    """
    authors: List[Author] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        authors = list(pool.map(lambda p: get_complete_poems_by_author(p[0]), poets))

    return [poem for author in authors for poem in author.poems]

def build_authors_database(
    poets: List[Tuple[str, int]],
    max_workers: int = 4
) -> List[Author]:
    """
    Fetch full Author objects—including metadata and poems—in parallel.

    Args:
        poets: List of (author_name, poem_count).
        max_workers: Number of threads to parallelize author fetching.

    Returns:
        A list of Author instances.
    """
    # 1. Load already-scraped authors
    existing_authors = load_author_checkpoints()
    existing_names = {author.name for author in existing_authors}

    # 2. Identify which poets remain to be processed
    to_process = [name for name, _ in poets if name not in existing_names]

    # 3. Scrape and checkpoint each missing author
    def _scrape_and_checkpoint(author_name: str) -> Author:
        author = get_complete_poems_by_author(author_name)
        save_author_checkpoint(author)
        return author

    new_authors: List[Author] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for author in pool.map(_scrape_and_checkpoint, to_process):
            new_authors.append(author)

    # 4. Return the union of existing and newly scraped
    return existing_authors + new_authors

