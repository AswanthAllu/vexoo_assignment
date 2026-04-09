"""
main.py
-------
Entry point for the Vexoo AI Assignment.

Runs both pipelines:
  1. Document ingestion + retrieval demo (Part 1)
  2. GSM8K simulated training pipeline  (Part 2)
  3. Bonus: Reasoning-Aware Query Router

Usage:
    python main.py                    # full demo
    python main.py --part ingestion   # ingestion only
    python main.py --part training    # training only
    python main.py --query "your question here"  # interactive retrieval
"""

import argparse
import sys

from utils.helpers import get_logger

logger = get_logger("main")

# ---------------------------------------------------------------------------
# Bonus: Reasoning-Aware Router
# ---------------------------------------------------------------------------

# Keyword sets driving the router — easily extensible
MATH_KEYWORDS = {
    "calculate", "solve", "equation", "integral", "derivative",
    "sum", "product", "algebra", "geometry", "measure", "percent",
    "how many", "how much", "profit", "loss", "average", "mean",
}

LEGAL_KEYWORDS = {
    "law", "legal", "court", "statute", "regulation", "contract",
    "liability", "clause", "constitute", "plaintiff", "defendant",
    "jurisdiction", "rights", "violation", "compliance", "sue",
}


def reasoning_router(query: str) -> str:
    """
    Route a query to the appropriate reasoning module.

    We use a fast token-based routing system here. It's incredibly quick 
    and handles basic logic perfectly. In a massive production system you 
    might swap this for a trained intent classifier, but the interface 
    would remain exactly the same.

    Routing Rules (evaluated in order of priority):
        1. Any math keyword present   → "math_module"
        2. Any legal keyword present  → "legal_module"
        3. Default                    → "general_retrieval"

    Args:
        query : The raw user query string.

    Returns:
        Module name string: "math_module" | "legal_module" | "general_retrieval"
    """
    lower_query = query.lower()

    # Check math signals — evaluated first (highest priority)
    if any(kw in lower_query for kw in MATH_KEYWORDS):
        return "math_module"

    # Check legal signals
    if any(kw in lower_query for kw in LEGAL_KEYWORDS):
        return "legal_module"

    # Default: general knowledge retrieval
    return "general_retrieval"


# ---------------------------------------------------------------------------
# Demo Document
# ---------------------------------------------------------------------------

DEMO_DOCUMENT = """
Artificial intelligence (AI) is transforming industries at an unprecedented pace.
Machine learning algorithms can now detect diseases with accuracy rivalling trained physicians.
Neural networks power real-time language translation across hundreds of languages.

In the legal domain, AI-assisted contract analysis tools can review thousands of clauses
in minutes, flagging compliance violations and reducing liability risk.
Courts in several jurisdictions have begun accepting AI-assisted evidence analysis,
raising questions about jurisdiction, regulation, and defendant rights.

Mathematics underpins every modern AI system.
Gradient descent algorithms minimise a loss function by computing derivatives and updating
model weights iteratively. Linear algebra — specifically matrix products — forms the
backbone of neural network forward passes. Statistical methods help evaluate model
accuracy using metrics like mean squared error and cross-entropy loss.

Climate science uses machine learning to improve weather prediction models.
Historical climate patterns are fed into deep learning architectures to forecast
temperature, precipitation, and extreme weather events.
These models run on distributed compute clusters optimised for parallel matrix operations.

Healthcare researchers are leveraging NLP to extract insights from clinical notes.
Named-entity recognition (NER) systems identify drug names, symptoms, and diagnoses
from unstructured text, speeding up clinical trials and drug discovery pipelines.
""".strip()


# ---------------------------------------------------------------------------
# Part 1: Ingestion + Retrieval Demo
# ---------------------------------------------------------------------------

def run_ingestion_demo(query: str = None) -> None:
    """Run the document ingestion and retrieval pipeline."""
    from ingestion.sliding_window import sliding_window_chunks
    from ingestion.knowledge_pyramid import build_pyramids_from_chunks
    from ingestion.retrieval import retrieve, print_result

    logger.info("=" * 60)
    logger.info("  Part 1: Document Ingestion + Retrieval")
    logger.info("=" * 60)

    # Step 1: Chunk the document
    chunks = sliding_window_chunks(DEMO_DOCUMENT, window_size=400, overlap=60)
    logger.info(f"Document split into {len(chunks)} chunks.")

    # Step 2: Build Knowledge Pyramids
    pyramids = build_pyramids_from_chunks(chunks)
    logger.info(f"Built {len(pyramids)} knowledge pyramids.")

    # Step 3: Log a sample pyramid
    if pyramids:
        sample = pyramids[0]
        logger.info(f"Sample Pyramid[0]  =>  Category: {sample['category']}  |  "
                    f"Keywords: {sample['keywords'][:5]}")

    # Step 4: Retrieval
    demo_queries = [
        query or "neural network machine learning model",
        "legal contract liability regulation",
        "calculate gradient loss derivatives",
    ]

    for q in demo_queries:
        # Route first
        route = reasoning_router(q)
        print(f"\n[Router] Query: {q!r}  =>  Module: {route}")

        # Then retrieve
        results = retrieve(q, pyramids, top_k=1)
        print_result(q, results)


# ---------------------------------------------------------------------------
# Part 2: Training Pipeline Demo
# ---------------------------------------------------------------------------

def run_training_demo() -> None:
    """Run the simulated GSM8K training pipeline."""
    from training.train_gsm8k import run_pipeline

    # Use a small subset so the demo completes in seconds
    run_pipeline(train_size=300, eval_size=100, epochs=3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vexoo AI Assignment — Document Ingestion & Training Pipeline"
    )
    parser.add_argument(
        "--part",
        choices=["ingestion", "training", "all"],
        default="all",
        help="Which part to run (default: all)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom retrieval query for the ingestion demo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.part in ("ingestion", "all"):
        run_ingestion_demo(query=args.query)

    if args.part in ("training", "all"):
        run_training_demo()

    # Quick router demo when running full suite
    if args.part == "all":
        print("\n" + "=" * 60)
        print("  Bonus: Reasoning-Aware Router Demonstrations")
        print("=" * 60)
        test_queries = [
            "Solve the equation 3x + 5 = 20",
            "Is this contract clause legally compliant?",
            "What is the history of deep learning?",
        ]
        for tq in test_queries:
            route = reasoning_router(tq)
            print(f"  Query : {tq!r}")
            print(f"  Route : {route}\n")


if __name__ == "__main__":
    main()
