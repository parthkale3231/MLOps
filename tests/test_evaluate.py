"""Unit tests for model evaluation and quality gate."""

import pytest
from src.evaluate import evaluate


def test_evaluate_execution():
    """Verify evaluate() computes classification metrics on test partition."""
    metrics = evaluate(use_test_split=True, min_f1=0.50)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["f1"] >= 0.50


def test_evaluate_quality_gate_failure():
    """Verify evaluate() raises RuntimeError when quality gate fails."""
    with pytest.raises(RuntimeError, match="quality gate failed"):
        evaluate(use_test_split=True, min_f1=0.99)
