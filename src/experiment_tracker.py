"""
MLflow experiment tracker for LLM evaluations.

MLflow tracks four things per experiment run:
- Parameters: what you changed (model, prompt_version, chunk_size, k)
- Metrics: what resulted (faithfulness, answer_relevancy, etc.)
- Artifacts: files produced (eval results CSV, comparison charts)
- Tags: metadata (git commit, environment, dataset size)

This gives a complete audit trail: for any score you see in the MLflow UI,
you can trace back to exactly which prompt version, chunk size, and model
produced it. Critical for understanding regressions.

Run `mlflow ui` to open the web dashboard at http://localhost:5000.
"""

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "rag-llmops-evaluation"
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def setup_mlflow(tracking_uri: str = "./mlruns") -> None:
    """
    Configure MLflow to use local file storage.

    For production: point to a remote MLflow tracking server or Databricks.
    For this project: local ./mlruns directory is sufficient.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow configured: tracking at '{tracking_uri}', experiment '{EXPERIMENT_NAME}'")
    logger.info("Run `mlflow ui` to open dashboard at http://localhost:5000")


def log_eval_run(
    eval_df: pd.DataFrame,
    params: Dict[str, Any],
    run_name: str,
    results_path: Optional[str] = None,
) -> str:
    """
    Log one evaluation run to MLflow.

    Args:
        eval_df: DataFrame with RAGAS metric columns
        params: Hyperparameters for this run (model, prompt_version, chunk_size, k)
        run_name: Human-readable label (e.g. "v1_baseline", "v2_tighter_prompt")
        results_path: Optional path to CSV artifact to attach

    Returns:
        run_id for cross-referencing runs
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log what you changed (parameters)
        mlflow.log_params(params)

        # Log aggregate metrics
        for metric in METRICS:
            if metric in eval_df.columns:
                mlflow.log_metric(f"mean_{metric}", eval_df[metric].mean())
                mlflow.log_metric(f"min_{metric}", eval_df[metric].min())
                mlflow.log_metric(f"std_{metric}", eval_df[metric].std())

        # Log the full per-question results as a downloadable artifact
        if results_path and os.path.exists(results_path):
            mlflow.log_artifact(results_path)

        mlflow.set_tags({
            "dataset_size": len(eval_df),
            "environment": "development",
        })

        run_id = run.info.run_id
        logger.info(f"MLflow run '{run_name}' logged: {run_id}")
        return run_id


def compare_runs_in_mlflow(run_id_a: str, run_id_b: str) -> Dict[str, Dict]:
    """
    Compare two runs from MLflow — returns delta for each metric.

    Useful for programmatic comparisons in CI or in notebooks.
    For visual comparisons, use the MLflow UI directly.
    """
    client = mlflow.tracking.MlflowClient()
    run_a = client.get_run(run_id_a)
    run_b = client.get_run(run_id_b)

    comparison = {}
    for metric in METRICS:
        key = f"mean_{metric}"
        val_a = run_a.data.metrics.get(key, 0)
        val_b = run_b.data.metrics.get(key, 0)
        delta = val_b - val_a
        comparison[metric] = {
            "run_a": round(val_a, 3),
            "run_b": round(val_b, 3),
            "delta": round(delta, 3),
            "winner": "B" if delta > 0 else ("A" if delta < 0 else "tie"),
        }

    return comparison
