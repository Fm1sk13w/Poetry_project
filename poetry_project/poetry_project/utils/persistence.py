import pickle
from pathlib import Path
from typing import List

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
