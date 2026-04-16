"""
CLI script – run RAGAS evaluation on the RAG system.

Usage
-----
    # Evaluate the live Cloud Run deployment (recommended for CI / staging checks):
    RAG_API_URL=https://api-rag-82274106778.us-central1.run.app python scripts/run_evals.py

    # Evaluate locally (direct Python import, useful offline / during development):
    python scripts/run_evals.py

Reads:  data/eval_dataset.json
Writes: data/eval_results_<UTC-timestamp>.json

Environment variables
---------------------
RAG_API_URL   Base URL of the deployed service (no trailing slash).
              When set, answers come from POST <RAG_API_URL>/chat so the
              evaluation targets the actual Cloud Run container.
              When absent, ask_question() is imported directly from chat.py.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable

# Make the repo root importable regardless of where the script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.evaluation import METRIC_NAMES, run_evaluation  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(_REPO_ROOT, "data", "eval_dataset.json")
RESULTS_DIR = os.path.join(_REPO_ROOT, "data")


# ---------------------------------------------------------------------------
# Answer function – Cloud Run HTTP or local import
# ---------------------------------------------------------------------------

def _make_remote_answer_fn(api_url: str) -> Callable[[str], str]:
    """Return a callable that hits the live /chat endpoint on Cloud Run."""
    import httpx  # already in requirements.txt

    base = api_url.rstrip("/")

    def _call(question: str) -> str:
        response = httpx.post(
            f"{base}/chat",
            json={"question": question},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["answer"]

    return _call


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    norm = os.path.normpath(path)
    if not os.path.exists(norm):
        sys.exit(f"ERROR: dataset not found at {norm}")
    with open(norm, encoding="utf-8") as f:
        return json.load(f)


def save_results(results: dict, results_dir: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"eval_results_{timestamp}.json"
    path = os.path.normpath(os.path.join(results_dir, filename))
    payload = {
        "timestamp": timestamp,
        "num_samples": len(results["per_question"]),
        "scores": results["scores"],
        "overall": results["overall"],
        "per_question": results["per_question"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_WIDTH = 64


def _score_bar(score: float | None, width: int = 12) -> str:
    if score is None:
        return "[" + "?" * width + "]"
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def print_report(results: dict, mode: str) -> None:
    sep = "=" * _WIDTH
    thin = "─" * _WIDTH

    print()
    print(sep)
    print("  RAGAS EVALUATION REPORT")
    print(sep)
    print(f"  Mode              : {mode}")
    print(f"  Samples evaluated : {len(results['per_question'])}")
    print(thin)
    print(f"  {'Metric':<30} {'Score':>6}  Visual")
    print(thin)

    for name in METRIC_NAMES:
        score = results["scores"].get(name)
        label = name.replace("_", " ").title()
        bar = _score_bar(score)
        score_str = f"{score:.4f}" if score is not None else "  N/A "
        print(f"  {label:<30} {score_str}  {bar}")

    print(thin)
    overall_bar = _score_bar(results["overall"])
    print(f"  {'Overall':<30} {results['overall']:.4f}  {overall_bar}")
    print(sep)

    print()
    print(thin)
    print("  PER-QUESTION BREAKDOWN")
    print(thin)

    for i, item in enumerate(results["per_question"], 1):
        q = item["question"]
        a = item["answer"].strip()
        gt = item["ground_truth"].strip()
        per_scores = item["scores"]

        print(f"\n  [{i}] Question : {q}")
        print(f"      Answer   : {a[:100]}{'...' if len(a) > 100 else ''}")
        print(f"      Expected : {gt[:100]}{'...' if len(gt) > 100 else ''}")

        score_parts = []
        for name in METRIC_NAMES:
            v = per_scores.get(name)
            short = name[:6]
            score_parts.append(f"{short}={v:.3f}" if v is not None else f"{short}=N/A")
        print("      Scores   : " + "  ".join(score_parts))

    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_url = os.getenv("RAG_API_URL", "").strip()

    if api_url:
        mode = f"remote  →  {api_url}"
        answer_fn = _make_remote_answer_fn(api_url)
        print(f"\nTarget: Cloud Run  ({api_url})")
    else:
        mode = "local (direct import)"
        answer_fn = None  # evaluation.py will import ask_question lazily
        print("\nTarget: local (no RAG_API_URL set — importing ask_question directly)")

    print(f"Loading evaluation dataset from: {DATASET_PATH}")
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} samples.\n")

    print("Running RAG pipeline + RAGAS scoring (may take 1-3 minutes)...\n")
    results = run_evaluation(dataset, answer_fn=answer_fn)

    print_report(results, mode)

    output_path = save_results(results, RESULTS_DIR)
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
