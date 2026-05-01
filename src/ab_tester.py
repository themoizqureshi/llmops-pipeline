"""
A/B testing harness for RAG configuration comparison.

The cardinal rule: change ONE variable at a time.
- Testing model A vs model B? Keep prompt, chunk_size, k identical.
- Testing prompt v1 vs v2? Keep model, chunk_size, k identical.

If you change two variables simultaneously, you cannot attribute the
score difference to either one — the experiment is meaningless.

What to A/B test (in order of impact):
1. System prompt (often the biggest lever — see Project 2)
2. Chunk size: smaller = more precise retrieval, larger = more context per chunk
3. k (retrieved chunks): more = higher recall, lower precision
4. Embedding model: HuggingFace local vs Google API quality trade-off
5. LLM model: Gemini 2.0 Flash vs a stronger model for complex reasoning
"""

import logging
import time
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from .experiment_tracker import log_eval_run, setup_mlflow

logger = logging.getLogger(__name__)

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def run_ab_test(
    variant_a_config: Dict[str, Any],
    variant_b_config: Dict[str, Any],
    qa_pairs: List[Dict],
    chain_builder_fn: Callable,
    evaluator_fn: Callable,
    tracking_uri: str = "./mlruns",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run a controlled A/B test between two RAG configurations.

    Args:
        variant_a_config: Config dict for the baseline (e.g. {"prompt_version": "v1", "k": 4})
        variant_b_config: Config dict for the challenger (e.g. {"prompt_version": "v2", "k": 4})
        qa_pairs: Evaluation Q&A dataset
        chain_builder_fn: Function(config) -> (chain, retriever)
        evaluator_fn: Function(chain, retriever, qa_pairs) -> pd.DataFrame with RAGAS columns

    Returns:
        (results_a, results_b, comparison_dict)
    """
    setup_mlflow(tracking_uri)

    logger.info("--- Running Variant A (baseline) ---")
    start_a = time.time()
    chain_a, retriever_a = chain_builder_fn(**variant_a_config)
    results_a = evaluator_fn(chain_a, retriever_a, qa_pairs)
    results_a["latency_seconds"] = time.time() - start_a
    run_id_a = log_eval_run(results_a, variant_a_config, "variant_a_baseline")

    logger.info("--- Running Variant B (challenger) ---")
    start_b = time.time()
    chain_b, retriever_b = chain_builder_fn(**variant_b_config)
    results_b = evaluator_fn(chain_b, retriever_b, qa_pairs)
    results_b["latency_seconds"] = time.time() - start_b
    run_id_b = log_eval_run(results_b, variant_b_config, "variant_b_challenger")

    # Build comparison dict
    comparison = {}
    for metric in METRICS:
        if metric not in results_a.columns or metric not in results_b.columns:
            continue
        a_mean = results_a[metric].mean()
        b_mean = results_b[metric].mean()
        delta = b_mean - a_mean
        comparison[metric] = {
            "variant_a": round(a_mean, 3),
            "variant_b": round(b_mean, 3),
            "delta": round(delta, 3),
            "delta_pct": round((delta / a_mean) * 100, 1) if a_mean > 0 else 0,
            "winner": "B" if delta > 0.01 else ("A" if delta < -0.01 else "tie"),
        }

    _print_ab_summary(variant_a_config, variant_b_config, comparison)
    return results_a, results_b, comparison


def _print_ab_summary(config_a: Dict, config_b: Dict, comparison: Dict) -> None:
    """Print a formatted terminal comparison table."""
    print("\n" + "=" * 65)
    print("A/B TEST RESULTS")
    print(f"  A (baseline):   {config_a}")
    print(f"  B (challenger): {config_b}")
    print("=" * 65)
    for metric, v in comparison.items():
        winner_icon = "🏆" if v["winner"] == "B" else ("  " if v["winner"] == "A" else "➖")
        print(
            f"{winner_icon} {metric:25s}: "
            f"A={v['variant_a']:.3f} | B={v['variant_b']:.3f} | "
            f"Δ={v['delta']:+.3f} ({v['delta_pct']:+.1f}%)"
        )
    print("=" * 65)
    winners = [v["winner"] for v in comparison.values()]
    b_wins = winners.count("B")
    a_wins = winners.count("A")
    print(f"Overall: B wins {b_wins}/{len(winners)} metrics, A wins {a_wins}/{len(winners)} metrics")
