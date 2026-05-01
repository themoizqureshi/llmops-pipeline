"""
Regression checker — the CI/CD quality gate.

This script runs in GitHub Actions on every push to main that touches
src/ or prompts/. It loads the most recent eval results CSV and compares
each metric against defined thresholds. If any metric is below threshold,
the script exits with code 1 — failing the CI build and blocking the merge.

Threshold philosophy:
- Start conservative (0.60) and raise as the system improves
- Set thresholds at ~0.05 below your current best score
- Tight enough to catch genuine regressions, loose enough to tolerate
  natural run-to-run variance in LLM-as-judge scoring
- Never lower a threshold once it's been established in production
"""

import glob
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Quality gates — CI fails if any metric drops below these
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_recall": 0.65,
    "context_precision": 0.65,
}


def check_regression(eval_results_path: str) -> bool:
    """
    Load eval results CSV and check each metric against thresholds.

    Returns True (all pass) or False (at least one failure).
    Exits with code 1 on failure so CI detects it.
    """
    df = pd.read_csv(eval_results_path)

    print("\n" + "=" * 60)
    print("REGRESSION CHECK")
    print(f"Results file: {eval_results_path}")
    print(f"Questions evaluated: {len(df)}")
    print("=" * 60)

    all_pass = True
    for metric, threshold in THRESHOLDS.items():
        if metric not in df.columns:
            logger.warning(f"  ⚠️  {metric}: not found in results (skipping)")
            continue

        actual = df[metric].mean()
        passed = actual >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"
        gap = actual - threshold

        print(f"  {status}  {metric:25s}: {actual:.3f}  (threshold: {threshold:.3f}, gap: {gap:+.3f})")

        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("✅ All quality gates passed. Safe to deploy.")
    else:
        print("❌ Quality regression detected. Check your prompt or code changes.")
        print("   Tip: run the eval locally and inspect LangSmith traces for failing questions.")
    print()

    return all_pass


def find_latest_results() -> str:
    """Find the most recently written results CSV."""
    pattern = "results/*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No eval results found matching '{pattern}'. Run evaluation first.")
        sys.exit(1)
    return files[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG regression quality gate")
    parser.add_argument("--results", default=None, help="Path to eval results CSV (default: latest in results/)")
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    logger.info(f"Checking: {results_path}")

    passed = check_regression(results_path)
    sys.exit(0 if passed else 1)
