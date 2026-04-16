"""
Chunking strategy comparison script.

Ingests the same PDF with each ChunkingStrategy, runs RAGAS evaluation
on each, and produces a comparison table + JSON report.

Usage
-----
    # Uses the first PDF found in data/ and eval_dataset.json:
    python scripts/compare_chunking.py

    # Specify a PDF explicitly:
    python scripts/compare_chunking.py --pdf data/my_document.pdf

    # Keep temporary Qdrant collections after the run (for inspection):
    python scripts/compare_chunking.py --no-cleanup

Writes: data/chunking_comparison.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

# Repo root on sys.path so `app` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402
from app.services.evaluation import METRIC_NAMES, run_evaluation  # noqa: E402
from app.services.ingestion import ChunkingStrategy, ingest_file  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(_REPO_ROOT, "data", "eval_dataset.json")
RESULTS_PATH = os.path.join(_REPO_ROOT, "data", "chunking_comparison.json")

STRATEGIES = [ChunkingStrategy.FIXED, ChunkingStrategy.RECURSIVE, ChunkingStrategy.SEMANTIC]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pdf() -> str:
    """Return the first PDF found in data/, for use as a default."""
    pdfs = glob.glob(os.path.join(_REPO_ROOT, "data", "*.pdf"))
    if not pdfs:
        sys.exit("ERROR: no PDF found in data/. Use --pdf to specify one.")
    # Prefer the FINAL PRACTICAL WORK doc if present, otherwise take the first
    for p in pdfs:
        if "FINAL" in os.path.basename(p).upper():
            return p
    return pdfs[0]


def _load_dataset() -> list[dict]:
    norm = os.path.normpath(DATASET_PATH)
    if not os.path.exists(norm):
        sys.exit(f"ERROR: eval dataset not found at {norm}")
    with open(norm, encoding="utf-8") as f:
        return json.load(f)


def _temp_collection(strategy: ChunkingStrategy) -> str:
    """Temporary Qdrant collection name for a strategy run."""
    base = settings.QDRANT_COLLECTION_NAME
    return f"{base}_cmp_{strategy.value.lower()}"


def _delete_collections(names: list[str]) -> None:
    from qdrant_client import QdrantClient  # noqa: PLC0415

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    for name in names:
        try:
            client.delete_collection(name)
            print(f"   Deleted temp collection: {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"   Warning: could not delete {name}: {exc}")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "strategy":   10,
    "chunks":      7,
    "avg_size":    9,
    "faithfulness": 13,
    "answer_relevancy": 13,
    "context_precision": 17,
    "overall":    9,
}


def _row(*cells) -> str:
    widths = list(_COL_WIDTHS.values())
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(str(cell).center(w))
    return "| " + " | ".join(parts) + " |"


def print_comparison(rows: list[dict]) -> None:
    header = _row(
        "Strategy", "Chunks", "Avg Size",
        "Faithfulness", "Answer Rel.", "Ctx Precision", "Overall",
    )
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        s = r["scores"]
        print(_row(
            r["strategy"],
            r["ingest"]["chunks"],
            r["ingest"]["avg_chunk_size"],
            f"{s.get('faithfulness', 0):.3f}",
            f"{s.get('answer_relevancy', 0):.3f}",
            f"{s.get('context_precision', 0):.3f}",
            f"{r['overall']:.3f}",
        ))

    print(sep)

    # Highlight best strategy
    best = max(rows, key=lambda r: r["overall"])
    print(f"\n  Best overall: {best['strategy']}  (overall={best['overall']:.3f})\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies via RAGAS")
    parser.add_argument("--pdf", help="Path to PDF file to ingest (default: first PDF in data/)")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary Qdrant collections after the run",
    )
    args = parser.parse_args()

    pdf_path = args.pdf or _find_pdf()
    if not os.path.exists(pdf_path):
        sys.exit(f"ERROR: PDF not found: {pdf_path}")

    print(f"\nPDF          : {os.path.basename(pdf_path)}")
    print(f"Eval dataset : {DATASET_PATH}")
    print(f"Strategies   : {[s.value for s in STRATEGIES]}\n")

    dataset = _load_dataset()
    print(f"Loaded {len(dataset)} eval samples.\n")

    rows: list[dict] = []
    temp_collections: list[str] = []

    for strategy in STRATEGIES:
        coll = _temp_collection(strategy)
        temp_collections.append(coll)

        print(f"{'=' * 60}")
        print(f"  Strategy: {strategy.value}")
        print(f"{'=' * 60}")

        # --- Ingest ---
        print(f"\n[1/2] Ingesting with {strategy.value}...")
        try:
            ingest_stats = ingest_file(pdf_path, strategy=strategy, collection_name=coll)
        except ImportError as exc:
            print(f"  SKIP — {exc}")
            rows.append({
                "strategy": strategy.value,
                "ingest": {"chunks": "N/A", "avg_chunk_size": "N/A"},
                "scores": {n: None for n in METRIC_NAMES},
                "overall": 0.0,
                "skipped": True,
                "skip_reason": str(exc),
            })
            continue

        # --- Evaluate ---
        print(f"\n[2/2] Running RAGAS evaluation against collection '{coll}'...")
        eval_results = run_evaluation(dataset, collection_name=coll)

        rows.append({
            "strategy": strategy.value,
            "collection": coll,
            "ingest": ingest_stats,
            "scores": eval_results["scores"],
            "overall": eval_results["overall"],
            "per_question": eval_results["per_question"],
        })
        print(f"  Done. Overall RAGAS score: {eval_results['overall']:.4f}\n")

    # --- Print table ---
    print_comparison(rows)

    # --- Save JSON ---
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pdf": os.path.basename(pdf_path),
        "num_eval_samples": len(dataset),
        "results": rows,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Comparison saved to: {RESULTS_PATH}\n")

    # --- Cleanup ---
    if not args.no_cleanup:
        print("Cleaning up temporary Qdrant collections...")
        _delete_collections(temp_collections)
        print()
    else:
        print(f"Temporary collections kept: {temp_collections}\n")


if __name__ == "__main__":
    main()
