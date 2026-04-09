"""
helpers.py
----------
Shared utility functions used across the project to keep things DRY.
"""

import re
import math
import logging
from typing import List, Dict


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "vexoo") -> logging.Logger:
    """Return a pre-configured logger for consistent log formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract the most frequent meaningful words from text.
    We just use a simple frequency-based approach here because it's
    usually enough for keyword extraction without needing heavy NLP tools.

    Args:
        text  : Input string.
        top_n : Number of top keywords to return.

    Returns:
        List of keyword strings sorted by frequency (descending).
    """
    # Remove punctuation and lowercase
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Simple stop-word filter
    stop_words = {
        "the", "and", "for", "are", "was", "with", "this", "that",
        "have", "from", "but", "not", "you", "all", "can", "her",
        "his", "they", "she", "one", "our", "out", "use", "has",
        "been", "its", "also", "more", "will", "each", "than",
        "then", "into", "about", "what", "when", "which", "some",
    }
    filtered = [w for w in words if w not in stop_words]

    # Count frequencies
    freq: Dict[str, int] = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency descending and return top_n
    sorted_words = sorted(freq, key=lambda w: freq[w], reverse=True)
    return sorted_words[:top_n]


def first_n_sentences(text: str, n: int = 3) -> str:
    """
    Return the first N sentences of a text block.

    Used as a lightweight 'summary' placeholder without heavy NLP.

    Args:
        text : Input string.
        n    : Number of sentences to return.

    Returns:
        String of first n sentences joined together.
    """
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:n])


# ---------------------------------------------------------------------------
# Similarity Utilities
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two numeric vectors.

    Args:
        vec_a : First vector (list of floats).
        vec_b : Second vector (list of floats).

    Returns:
        Float in range [-1, 1]; 1 = identical direction.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of equal length.")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def text_to_bow_vector(text: str, vocab: List[str]) -> List[float]:
    """
    Convert text to a bag-of-words vector based on a given vocabulary.
    This is a nice lightweight alternative to using huge embeddings models for this assignment.

    Args:
        text  : Input string.
        vocab : Shared vocabulary list (union of all keywords).

    Returns:
        List of float counts (one per vocab term).
    """
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
    return [1.0 if term in words else 0.0 for term in vocab]
