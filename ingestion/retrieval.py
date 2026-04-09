"""
retrieval.py
------------
Handles retrieving relevant chunks by comparing the query against our Knowledge Pyramid.

We do multi-level retrieval here. We compare the query against all 4 layers of the pyramid.
Each layer contributes to the final score, so we get a really solid semantic match.
For simplicity without external ML libs, we use SequenceMatcher for raw text and 
bag-of-words cosine similarity for keywords.
"""

import difflib
from typing import List, Dict, Tuple

from utils.helpers import cosine_similarity, text_to_bow_vector, extract_keywords


# ---------------------------------------------------------------------------
# Layer-level Scorers
# ---------------------------------------------------------------------------

def _score_raw_text(query: str, raw_text: str) -> float:
    """SequenceMatcher ratio on the raw text layer (0.0 – 1.0)."""
    return difflib.SequenceMatcher(None, query.lower(), raw_text.lower()).ratio()


def _score_summary(query: str, summary: str) -> float:
    """SequenceMatcher ratio on the summary layer (0.0 – 1.0)."""
    return difflib.SequenceMatcher(None, query.lower(), summary.lower()).ratio()


def _score_category(query: str, category: str) -> float:
    """
    Bonus score if the category word appears in the query.
    Returns 0.5 for a match, 0.0 otherwise.
    We just use a fixed 0.5 bonus to keep the combined score balanced.
    """
    return 0.5 if category.lower() in query.lower() else 0.0


def _score_keywords(query: str, chunk_keywords: List[str]) -> float:
    """
    Cosine similarity between query keyword vector and chunk keyword vector.

    Both are projected onto the shared vocabulary of chunk keywords.
    This converts the keyword layer into a numeric similarity score.
    """
    if not chunk_keywords:
        return 0.0

    vocab = chunk_keywords  # use chunk's keywords as the vector space
    query_vec = text_to_bow_vector(query, vocab)
    chunk_vec = [1.0] * len(vocab)  # chunk already has these keywords

    return cosine_similarity(query_vec, chunk_vec)


# ---------------------------------------------------------------------------
# Combined Scorer
# ---------------------------------------------------------------------------

def _combined_score(
    query: str,
    pyramid: Dict[str, object],
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute a weighted combination of all 4 layer scores.

    Default weights give higher importance to keyword & summary matches —
    they're more semantically dense than raw text fuzzy matching.

    Args:
        query   : User's search query.
        pyramid : A single pyramid dict from build_pyramids_from_chunks.
        weights : Optional override of layer weights.

    Returns:
        A scalar relevance score (higher = more relevant).
    """
    if weights is None:
        weights = {
            "raw_text": 0.15,   # noisy, verbose — lowest weight
            "summary":  0.30,   # condensed meaning — medium weight
            "category": 0.20,   # category hint — medium weight
            "keywords": 0.35,   # semantic fingerprint — highest weight
        }

    score = 0.0
    score += weights["raw_text"] * _score_raw_text(query, pyramid["raw_text"])
    score += weights["summary"]  * _score_summary(query, pyramid["summary"])
    score += weights["category"] * _score_category(query, pyramid["category"])
    score += weights["keywords"] * _score_keywords(query, pyramid["keywords"])
    return score


# ---------------------------------------------------------------------------
# Public Retrieval API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    pyramids: List[Dict[str, object]],
    top_k: int = 1,
) -> List[Tuple[float, Dict[str, object]]]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query    : The user's search string.
        pyramids : List of pyramid dicts (entire knowledge base).
        top_k    : Number of top results to return (default 1).

    Returns:
        List of (score, pyramid) tuples sorted by descending score.
    """
    if not pyramids:
        return []

    scored = [
        (_combined_score(query, p), p)
        for p in pyramids
    ]

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def print_result(query: str, results: List[Tuple[float, Dict[str, object]]]) -> None:
    """Pretty-print retrieval results to stdout."""
    print(f"\n{'='*60}")
    print(f"  Query : {query!r}")
    print(f"{'='*60}")
    if not results:
        print("  No results found.")
        return

    for rank, (score, pyramid) in enumerate(results, start=1):
        print(f"\n  Rank #{rank}  |  Chunk ID: {pyramid['chunk_id']}  |  Score: {score:.4f}")
        print(f"  Category : {pyramid['category']}")
        print(f"  Keywords : {', '.join(pyramid['keywords'])}")
        print(f"  Summary  : {pyramid['summary'][:150]}...")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Stand-alone demo
    from ingestion.knowledge_pyramid import build_pyramids_from_chunks
    from ingestion.sliding_window import sliding_window_chunks

    doc = (
        "Deep learning is revolutionizing artificial intelligence. "
        "Neural networks with many layers can learn hierarchical representations. "
        "Applications include image classification, language translation, and more. "
    ) * 60

    chunks = sliding_window_chunks(doc)
    pyramids = build_pyramids_from_chunks(chunks)

    results = retrieve("neural network image classification", pyramids, top_k=2)
    print_result("neural network image classification", results)
