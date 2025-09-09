"""
utils/lexical_diversity.py

Provides functions for computing length‑robust measures of an author's lexical
breadth (vocabulary diversity) from their collected works.

This module implements several established metrics from computational
linguistics and stylometry, including:

- MTLD (Measure of Textual Lexical Diversity): A length‑insensitive measure
  that calculates the mean length of word sequences maintaining a given
  type–token ratio threshold.
- HD‑D (Hypergeometric Distribution Diversity): Estimates the probability
  of encountering new word types in a fixed‑size random sample, reducing
  bias from text length.
- Fixed‑budget vocabulary size: Estimates the expected number of unique
  word types in a random sample of N tokens, averaged over multiple trials.

All functions support consistent token preprocessing via spaCy, with options
to lowercase, lemmatize, and remove punctuation, numbers, stopwords, or
proper nouns. This ensures comparability across authors and corpora.

Typical usage involves:
    1. Loading Author objects with their poems.
    2. Concatenating each author's poems into a single token stream.
    3. Passing the token stream through one of the lexical diversity
       functions to obtain a score.
    4. Integrating the resulting scores into the project's metrics DataFrame.

Example:
    >>> from utils.lexical_diversity import compute_author_lexical_breadth
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    >>> scores = compute_author_lexical_breadth(authors, nlp, method="mtld")
    >>> scores["William Shakespeare"]
    87.45

These metrics are designed to be robust to differences in poem count and
length, making them suitable for fair cross‑author comparisons in literary
analysis.

"""

import math
import random
from collections import Counter
from typing import Iterable, List, Dict, Callable, Tuple, Optional

def _clean_tokens_spacy(doc, *, lowercase=True, lemmatize=True,
                        remove_punct=True, remove_numbers=True,
                        remove_stopwords=False, remove_propn=False) -> List[str]:
    tokens = []
    for t in doc:
        if remove_punct and (t.is_punct or t.is_space):
            continue
        if remove_numbers and (t.like_num or t.is_currency):
            continue
        if remove_stopwords and t.is_stop:
            continue
        if remove_propn and t.pos_ == "PROPN":
            continue
        form = t.lemma_ if lemmatize else t.text
        if lowercase:
            form = form.lower()
        if form and form.isalpha():
            tokens.append(form)
    return tokens

# 1) MTLD (McCarthy & Jarvis, 2010)
def mtld(tokens: List[str], ttr_threshold: float = 0.72, min_segment: int = 10) -> float:
    if len(tokens) < min_segment:
        return float("nan")

    def _mtld_seq(seq):
        factors = 0
        types = set()
        token_count = 0
        for tok in seq:
            token_count += 1
            types.add(tok)
            ttr = len(types) / token_count
            if token_count >= min_segment and ttr <= ttr_threshold:
                factors += 1
                types.clear()
                token_count = 0
        # partial factor
        if token_count > 0:
            factors += (1 - (len(types) / token_count)) / (1 - ttr_threshold) if (1 - ttr_threshold) > 0 else 0
        return len(seq) / factors if factors > 0 else float("nan")

    forward = _mtld_seq(tokens)
    backward = _mtld_seq(list(reversed(tokens)))
    return (forward + backward) / 2 if math.isfinite(forward) and math.isfinite(backward) else forward

# 2) HD-D (D for Diversity; uses a hypergeometric expectation)
def hdd(tokens: List[str], sample_size: int = 42) -> float:
    N = len(tokens)
    if N == 0:
        return float("nan")
    counts = Counter(tokens)
    # Expected contribution of each type to TTR in a sample of size s
    s = min(sample_size, N)
    hdd_value = 0.0
    for freq in counts.values():
        # Probability that type occurs at least once in a sample of size s without replacement
        # P(X>=1) = 1 - C(N - freq, s) / C(N, s)
        # use logs for stability
        def logC(n, k):
            if k < 0 or k > n:
                return float("-inf")
            return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        p_at_least_one = 1.0 - math.exp(logC(N - freq, s) - logC(N, s))
        hdd_value += p_at_least_one / s
    return hdd_value

# 3) Fixed-budget vocabulary: expected unique types at N tokens via subsampling
def vocab_at_budget(tokens: List[str], budget: int = 1000, trials: int = 200, seed: int = 42) -> float:
    if len(tokens) == 0:
        return float("nan")
    rng = random.Random(seed)
    m = min(budget, len(tokens))
    uniq_counts = 0
    for _ in range(trials):
        sample = rng.sample(tokens, m) if m < len(tokens) else list(tokens)
        uniq_counts += len(set(sample))
    return uniq_counts / trials

# High-level API
def compute_author_lexical_breadth(
    authors,
    nlp,  # spaCy pipeline
    method: str = "mtld",  # "mtld" | "hdd" | "budget"
    *,
    budget_tokens: int = 1000,
    trials: int = 200,
    preprocess_kwargs: Optional[dict] = None,
    min_tokens: int = 300
) -> Dict[str, float]:
    """
    Compute a length-robust lexical breadth score per author.

    Args:
        authors: Iterable of Author objects with .name and .poems (with .content).
        nlp: spaCy language model.
        method: 'mtld' (default), 'hdd', or 'budget'.
        budget_tokens: Target tokens for 'budget' method (ignored otherwise).
        trials: Subsampling trials for 'budget' method.
        preprocess_kwargs: Dict of token cleaning options for _clean_tokens_spacy.
        min_tokens: Minimum tokens required for a valid score.

    Returns:
        Mapping: author_name -> lexical breadth score (float or NaN if insufficient data).
    """
    preprocess_kwargs = preprocess_kwargs or dict(
        lowercase=True, lemmatize=True, remove_punct=True, remove_numbers=True,
        remove_stopwords=False, remove_propn=False
    )

    results: Dict[str, float] = {}
    for author in authors:
        # Concatenate all poems
        text = "\n".join(p.content for p in author.poems if getattr(p, "content", None))
        if not text.strip():
            results[author.name] = float("nan")
            continue
        doc = nlp(text)
        tokens = _clean_tokens_spacy(doc, **preprocess_kwargs)

        if len(tokens) < min_tokens:
            results[author.name] = float("nan")
            continue

        if method == "mtld":
            score = mtld(tokens)
        elif method == "hdd":
            score = hdd(tokens)
        elif method == "budget":
            score = vocab_at_budget(tokens, budget=budget_tokens, trials=trials)
        else:
            raise ValueError("method must be one of: 'mtld', 'hdd', 'budget'")

        results[author.name] = score
    return results
