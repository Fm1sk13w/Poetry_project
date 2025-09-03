"""
Pipeline module for scraping, persisting, and analyzing poetry data.

This module provides high-level functions to:
1. Retrieve poets within a specified poem count range.
2. Scrape poems and metadata for those poets.
3. Persist the complete authors database and derived metrics to disk.
4. Load persisted data for analysis without re-scraping.

Author: PoetryProjectBot
"""
from __future__ import annotations
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from poetry_project.models import Author, Poem
from poetry_project.scraper.author_scraper import (
    get_poets_by_poem_range,
    get_poem_urls_by_author,
)
from poetry_project.scraper.poem_scraper import scrape_poem_content
from poetry_project.metadata.wikidata import get_birth_and_nationality_wikidata
from poetry_project.utils.persistence import (
    load_author_checkpoints,
    save_author_checkpoint,
)
from poetry_project.utils.analysis_utils import compute_author_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

AUTHORS_DB_PATH = DATA_DIR / "authors.pkl"
METRICS_PATH = DATA_DIR / "author_metrics.parquet"


# ---------------------------------------------------------------------------
# Core scraping functions
# ---------------------------------------------------------------------------

def get_complete_poems_by_author(
    author_name: str,
    max_workers: int = 10
) -> Author:
    """
    Build an Author object populated with all poems and metadata,
    with progress feedback.
    """
    birth_year, nationality = get_birth_and_nationality_wikidata(author_name)
    author = Author(name=author_name, birth_year=birth_year, nationality=nationality)

    urls = get_poem_urls_by_author(author_name)
    print(f"    Found {len(urls)} poems for {author_name}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for _ in tqdm(pool.map(lambda u: scrape_poem_content(u, author), urls),
                      total=len(urls),
                      desc=f"    Scraping poems for {author_name}"):
            pass

    return author

def build_authors_database(
    poets: List[Tuple[str, int]],
    max_workers: int = 4
) -> List[Author]:
    """
    Fetch full Author objects—including metadata and poems—in parallel,
    with progress feedback.
    """
    existing_authors = load_author_checkpoints()
    existing_names = {author.name for author in existing_authors}

    # Filter out already processed authors and "Anonymous"
    to_process = [
        name for name, _ in poets
        if name not in existing_names and name.strip().lower() != "anonymous"
    ]

    print(f"Found {len(to_process)} authors to scrape.")

    def _scrape_and_checkpoint(author_name: str) -> Author:
        print(f"  → Starting {author_name}...")
        author = get_complete_poems_by_author(author_name)
        print(f"    ✓ Finished {author_name} ({len(author.poems)} poems)")
        save_author_checkpoint(author)
        return author

    new_authors: List[Author] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for author in tqdm(pool.map(_scrape_and_checkpoint, to_process), total=len(to_process), desc="Authors processed"):
            new_authors.append(author)

    return existing_authors + new_authors



# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_authors_database(authors: List[Author], path: Path = AUTHORS_DB_PATH) -> None:
    """
    Persist the complete authors database to disk.

    Args:
        authors: List of Author instances to save.
        path: Destination file path.
    """
    with open(path, "wb") as file:
        pickle.dump(authors, file)


def load_authors_database(path: Path = AUTHORS_DB_PATH) -> List[Author]:
    """
    Load the complete authors database from disk.

    Args:
        path: Path to the saved authors database.

    Returns:
        List of Author instances.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def save_metrics(metrics_df: pd.DataFrame, path: Path = METRICS_PATH) -> None:
    """
    Save author metrics DataFrame to disk in Parquet format.

    Args:
        metrics_df: DataFrame containing author metrics.
        path: Destination file path.
    """
    metrics_df.to_parquet(path, index=False)


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    """
    Load author metrics DataFrame from disk.

    Args:
        path: Path to the saved metrics file.

    Returns:
        DataFrame containing author metrics.
    """
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def run_full_pipeline(
    min_poems: int = 10,
    max_poems: float = float("inf"),
    author_threads: int = 4,
    poem_threads: int = 10
) -> None:
    """
    Execute the full scraping and persistence pipeline.

    Args:
        min_poems: Minimum number of poems an author must have.
        max_poems: Maximum number of poems an author can have.
        author_threads: Number of threads for author-level parallelism.
        poem_threads: Number of threads for poem-level parallelism.
    """
    poets = get_poets_by_poem_range(min_poems, max_poems)
    print(poets)
    # Override get_complete_poems_by_author's default max_workers
    def _scrape_author(name: str) -> Author:
        return get_complete_poems_by_author(name, max_workers=poem_threads)

    existing_authors = load_author_checkpoints()
    existing_names = {author.name for author in existing_authors}
    to_process = [name for name, _ in poets if name not in existing_names]

    new_authors: List[Author] = []
    with ThreadPoolExecutor(max_workers=author_threads) as pool:
        for author in pool.map(_scrape_author, to_process):
            save_author_checkpoint(author)
            new_authors.append(author)

    all_authors = existing_authors + new_authors

    # Save full authors database
    save_authors_database(all_authors)

    # Compute and save metrics
    metrics_df = compute_author_metrics(all_authors)
    save_metrics(metrics_df)

    print(f"Pipeline completed. Saved {len(all_authors)} authors and metrics to '{DATA_DIR}'")
