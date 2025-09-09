import pickle
from pathlib import Path
from typing import List, Optional
import pandas as pd

from poetry_project.models import Author

# Directory to store per-author checkpoints
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def _author_checkpoint_path(name: str) -> Path:
    """
    Build a filesystem-safe path for the given author.
    """
    filename = name.lower().replace(" ", "_") + ".pkl"
    return CHECKPOINT_DIR / filename

def save_author_checkpoint(author: Author) -> None:
    """
    Persist a single Author instance to disk.
    """
    path = _author_checkpoint_path(author.name)
    with open(path, "wb") as f:
        pickle.dump(author, f)

def load_author_checkpoints() -> List[Author]:
    """
    Load all previously saved Author instances.
    """
    authors: List[Author] = []
    for path in CHECKPOINT_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, Author):
                    authors.append(loaded)
        except Exception:
            # Corrupted or incompatible pickle; skip
            continue
    return authors

def clean_authors_dataset(
    authors: List["Author"],
    n_bins: Optional[int] = None,
    min_authors_per_bin: Optional[int] = None
) -> List["Author"]:
    """
    Remove authors without poems, deduplicate by keeping the one with the most poems,
    and optionally filter out authors from time bins with fewer than a given number of authors.

    Args:
        authors: List of Author objects.
        n_bins: If provided, number of time bins to split by birth_year for filtering.
        min_authors_per_bin: If provided, minimum authors required per bin to keep it.

    Returns:
        Cleaned (and optionally filtered) list of Author objects.
    """
    # 1. Remove authors with no poems
    authors = [a for a in authors if a.poems]

    # 2. Deduplicate by author name, keeping the one with more poems
    unique_authors = {}
    for author in authors:
        if author.name not in unique_authors:
            unique_authors[author.name] = author
        else:
            if len(author.poems) > len(unique_authors[author.name].poems):
                unique_authors[author.name] = author

    cleaned_authors = list(unique_authors.values())

    # 3. Optional filtering by bin representation
    if n_bins is not None and min_authors_per_bin is not None:
        # Keep only authors with valid birth_year
        df = pd.DataFrame(
            [(a, int(a.birth_year)) for a in cleaned_authors if a.birth_year and str(a.birth_year).isdigit()],
            columns=["author_obj", "birth_year"]
        )

        if not df.empty:
            # Assign bins
            df["bin"] = pd.cut(df["birth_year"], bins=n_bins)

            # Count authors per bin
            bin_counts = df["bin"].value_counts()

            # Keep only bins meeting the threshold
            valid_bins = bin_counts[bin_counts >= min_authors_per_bin].index

            # Filter authors
            df = df[df["bin"].isin(valid_bins)]
            cleaned_authors = df["author_obj"].tolist()

    return cleaned_authors
