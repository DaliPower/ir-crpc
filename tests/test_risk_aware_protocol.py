from __future__ import annotations

import numpy as np
import pytest

from experiments.risk_aware_protocol import (
    aggregate_seed_results,
    reward_components,
    select_best_checkpoint,
    squared_voltage_violation,
)


def test_squared_voltage_violation_is_zero_inside_limits():
    voltage = np.array([0.95, 0.98, 1.00, 1.03, 1.05])

    penalty = squared_voltage_violation(voltage, 0.95, 1.05)

    np.testing.assert_allclose(penalty, np.zeros_like(voltage))


def test_squared_voltage_violation_grows_quadratically():
    voltage = np.array([1.06, 1.07, 0.94, 0.93])

    penalty = squared_voltage_violation(voltage, 0.95, 1.05)

    np.testing.assert_allclose(penalty, [0.0001, 0.0004, 0.0001, 0.0004])


def test_reward_components_include_explicit_violation_penalty():
    result = reward_components(
        np.array([1.00, 1.07]),
        np.array([0.2, -0.4]),
        v_lower=0.95,
        v_upper=1.05,
        voltage_weight=1.0,
        violation_weight=100.0,
        q_weight=0.1,
    )

    assert result["voltage_deviation_loss"] == pytest.approx(0.035)
    assert result["voltage_violation_loss"] == pytest.approx(0.02)
    assert result["q_loss"] == pytest.approx(0.3)
    assert result["q_loss_weighted"] == pytest.approx(0.03)
    assert result["total_loss"] == pytest.approx(0.085)
    assert result["reward"] == pytest.approx(-0.085)


def test_checkpoint_selection_uses_validation_metrics_not_test_metrics():
    records = [
        {
            "episode": 100,
            "path": "episode_100.pt",
            "validation_metrics": {
                "percentage_of_v_out_of_control": 0.02,
                "average_voltage_deviation": 0.01,
                "q_loss": 0.20,
            },
            "test_metrics": {"percentage_of_v_out_of_control": 0.00},
        },
        {
            "episode": 200,
            "path": "episode_200.pt",
            "validation_metrics": {
                "percentage_of_v_out_of_control": 0.01,
                "average_voltage_deviation": 0.02,
                "q_loss": 0.30,
            },
            "test_metrics": {"percentage_of_v_out_of_control": 0.50},
        },
    ]

    best = select_best_checkpoint(records)

    assert best["episode"] == 200


def test_checkpoint_selection_breaks_exact_metric_tie_with_earlier_episode():
    metrics = {
        "percentage_of_v_out_of_control": 0.01,
        "average_voltage_deviation": 0.01,
        "q_loss": 0.2,
    }

    best = select_best_checkpoint(
        [
            {"episode": 80, "validation_metrics": metrics},
            {"episode": 40, "validation_metrics": metrics},
        ]
    )

    assert best["episode"] == 40


def test_seed_aggregation_uses_sample_standard_deviation_and_keeps_records():
    records = [
        {"seed": 0, "violation_pct": 1.0, "q_loss": 0.2},
        {"seed": 1, "violation_pct": 2.0, "q_loss": 0.4},
        {"seed": 2, "violation_pct": 3.0, "q_loss": 0.6},
    ]

    summary = aggregate_seed_results(records, ["violation_pct", "q_loss"])

    assert summary["metrics"]["violation_pct"]["mean"] == pytest.approx(2.0)
    assert summary["metrics"]["violation_pct"]["std"] == pytest.approx(1.0)
    assert summary["metrics"]["q_loss"]["mean"] == pytest.approx(0.4)
    assert summary["metrics"]["q_loss"]["std"] == pytest.approx(0.2)
    assert summary["records"] == records


def test_protocol_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        squared_voltage_violation(np.array([1.0, np.nan]), 0.95, 1.05)
