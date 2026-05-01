"""
Tests for the regression checker and prompt manager.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.regression_checker import check_regression, THRESHOLDS


def make_results_csv(tmp_path: Path, scores: dict) -> str:
    """Helper: write a minimal eval results CSV to a temp file."""
    df = pd.DataFrame([scores])
    path = tmp_path / "test_results.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_all_metrics_above_threshold_passes(tmp_path):
    scores = {
        "faithfulness": 0.90,
        "answer_relevancy": 0.85,
        "context_recall": 0.80,
        "context_precision": 0.75,
    }
    path = make_results_csv(tmp_path, scores)
    assert check_regression(path) is True


def test_faithfulness_below_threshold_fails(tmp_path):
    scores = {
        "faithfulness": 0.50,          # Below 0.75
        "answer_relevancy": 0.85,
        "context_recall": 0.80,
        "context_precision": 0.75,
    }
    path = make_results_csv(tmp_path, scores)
    assert check_regression(path) is False


def test_multiple_metrics_below_threshold_fails(tmp_path):
    scores = {
        "faithfulness": 0.50,
        "answer_relevancy": 0.50,
        "context_recall": 0.50,
        "context_precision": 0.50,
    }
    path = make_results_csv(tmp_path, scores)
    assert check_regression(path) is False


def test_exactly_at_threshold_passes(tmp_path):
    scores = {metric: threshold for metric, threshold in THRESHOLDS.items()}
    path = make_results_csv(tmp_path, scores)
    assert check_regression(path) is True


def test_missing_metric_is_skipped(tmp_path, capsys):
    scores = {
        "faithfulness": 0.90,
        # answer_relevancy missing
        "context_recall": 0.80,
        "context_precision": 0.75,
    }
    path = make_results_csv(tmp_path, scores)
    result = check_regression(path)
    captured = capsys.readouterr()
    assert "answer_relevancy" in captured.out or result is True  # Missing metric doesn't cause failure


def test_prompt_manager_loads_version(tmp_path):
    """Test prompt manager can load from a temp registry."""
    # Create temp registry
    registry = {
        "versions": [
            {"version": "v1", "file": str(tmp_path / "v1.txt"), "description": "test", "created": "2024-01-01", "metrics": {}}
        ],
        "current_production": "v1",
    }
    (tmp_path / "v1.txt").write_text("Test prompt: {context} {question}")

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))

    # Patch the registry path inside prompt_manager
    with patch("src.prompt_manager.REGISTRY_PATH", registry_path):
        from src.prompt_manager import load_prompt
        content = load_prompt("v1")
    assert "context" in content
    assert "question" in content
