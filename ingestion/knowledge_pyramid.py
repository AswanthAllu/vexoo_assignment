"""
knowledge_pyramid.py
---------------------
Builds a 4-layer Knowledge Pyramid for a text chunk.

We structure our knowledge into different layers (from Raw Text up to Keywords)
because it helps with retrieval. Like how humans naturally read things—you scan
the summary first, check the category, then read the raw text if it looks relevant.
"""

from typing import Dict, List
from utils.helpers import extract_keywords, first_n_sentences


# ---------------------------------------------------------------------------
# Category Rules — easily extensible by adding new entries
# ---------------------------------------------------------------------------

CATEGORY_RULES: Dict[str, List[str]] = {
    "science":   ["experiment", "hypothesis", "research", "biology", "chemistry",
                  "physics", "laboratory", "molecule", "atom", "energy"],
    "technology":["software", "algorithm", "computer", "data", "network",
                  "programming", "api", "machine", "learning", "model",
                  "artificial", "intelligence", "deep", "neural"],
    "legal":     ["law", "court", "legal", "statute", "regulation", "contract",
                  "jurisdiction", "liability", "clause", "defendant", "plaintiff"],
    "math":      ["equation", "calculus", "algebra", "theorem", "proof",
                  "geometry", "formula", "derivative", "integral", "matrix",
                  "calculate", "solve", "number", "plus", "minus"],
    "history":   ["century", "war", "empire", "ancient", "civilization",
                  "revolution", "dynasty", "historical", "period", "era"],
    "finance":   ["market", "investment", "stock", "revenue", "profit",
                  "interest", "capital", "budget", "economy", "bank"],
    "health":    ["medicine", "disease", "treatment", "patient", "symptom",
                  "clinical", "diagnosis", "therapy", "hospital", "drug"],
}


def classify_category(text: str) -> str:
    """
    Classify text into a broad category by counting basic keywords.
    We just use a simple rule-based approach here because it's super fast
    and handles our basic categories well enough without needing an LLM.

    Args:
        text : The chunk text to classify.

    Returns:
        Category string (e.g., "technology") or "general" if no match.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_RULES}

    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            # Count occurrences of each keyword in the text
            scores[category] += text_lower.count(keyword)

    best_category = max(scores, key=lambda c: scores[c])

    # Only assign a category if at least one keyword was found
    return best_category if scores[best_category] > 0 else "general"


# ---------------------------------------------------------------------------
# Pyramid Builder
# ---------------------------------------------------------------------------

def build_pyramid(chunk: str, chunk_id: int = 0) -> Dict[str, object]:
    """
    Construct a 4-layer Knowledge Pyramid from a text chunk.

    Args:
        chunk    : The raw text segment.
        chunk_id : Index of the chunk in the document (for tracking).

    Returns:
        A dictionary with keys:
            - chunk_id      : int
            - raw_text      : str   (Layer 1)
            - summary       : str   (Layer 2)
            - category      : str   (Layer 3)
            - keywords      : list  (Layer 4)
    """
    return {
        "chunk_id": chunk_id,

        # Layer 1: Original chunk — preserved exactly
        "raw_text": chunk,

        # Layer 2: Summary — first 2-3 sentences as a lightweight placeholder
        #          In production this would call an LLM summariser.
        "summary": first_n_sentences(chunk, n=3),

        # Layer 3: Category — rule-based theme detection
        "category": classify_category(chunk),

        # Layer 4: Distilled knowledge — top keywords as a semantic fingerprint
        #          In production these would be replaced by dense embeddings.
        "keywords": extract_keywords(chunk, top_n=10),
    }


def build_pyramids_from_chunks(chunks: List[str]) -> List[Dict[str, object]]:
    """
    Build a pyramid for every chunk in the document.

    Args:
        chunks : List of text chunks (output of sliding_window_chunks).

    Returns:
        List of pyramid dictionaries, one per chunk.
    """
    return [build_pyramid(chunk, chunk_id=i) for i, chunk in enumerate(chunks)]


if __name__ == "__main__":
    # Quick demonstration
    sample_text = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without explicit programming. "
        "Deep learning uses neural networks with many layers to model complex patterns. "
        "Applications include image recognition, natural language processing, and more. "
    ) * 5

    pyramid = build_pyramid(sample_text, chunk_id=0)
    print("=== Knowledge Pyramid ===")
    print(f"Chunk ID  : {pyramid['chunk_id']}")
    print(f"Category  : {pyramid['category']}")
    print(f"Keywords  : {pyramid['keywords']}")
    print(f"Summary   : {pyramid['summary'][:120]}...")
