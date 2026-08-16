"""Tests for autoencoder evaluation and threshold calibration."""
from __future__ import annotations

import json

import numpy as np

from python.evaluation.evaluate import calibrate_threshold, evaluate
from python.training.train_model import train


def test_threshold_is_calibrated_from_healthy_validation(tmp_path):
    train(tmp_path, seed=42, epochs=35, batch_size=64, learning_rate=1.0e-3)

    calibration = calibrate_threshold(
        tmp_path / "anomaly_model.json",
        tmp_path / "preprocessing_parameters.json",
    )

    assert calibration["threshold"] > 0.0
    assert calibration["validation_count"] > 0.0
    assert calibration["normal_validation_mean_error"] >= 0.0
    assert calibration["normal_validation_p99_error"] >= calibration[
        "normal_validation_p95_error"
    ]
    assert calibration["normal_validation_max_error"] >= calibration[
        "normal_validation_p99_error"
    ]


def test_all_fault_scenarios_are_evaluated(tmp_path):
    train(tmp_path, seed=42, epochs=45, batch_size=64, learning_rate=1.0e-3)

    result = evaluate(
        tmp_path / "anomaly_model.json",
        tmp_path / "preprocessing_parameters.json",
        tmp_path / "model_metadata.json",
    )

    assert result["architecture"] == [80, 32, 8, 32, 80]
    assert result["parameter_count"] == 5784
    assert result["threshold"] > 0.0

    expected = {
        "normal",
        "temperature_drift",
        "vibration_fault",
        "current_fault",
        "multi_sensor_fault",
    }
    assert set(result["scenarios"]) == expected

    for metrics in result["scenarios"].values():
        assert metrics["window_count"] > 0.0
        assert metrics["mean_error"] >= 0.0
        assert metrics["max_error"] >= metrics["p99_error"]
        assert 0.0 <= metrics["anomaly_rate"] <= 1.0

    metadata = json.loads(
        (tmp_path / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["threshold_status"] == "calibrated"
    assert metadata["threshold"] == result["threshold"]
    assert set(metadata["evaluation"]) == expected


def test_threshold_calibration_is_deterministic(tmp_path):
    train(tmp_path, seed=42, epochs=35, batch_size=64, learning_rate=1.0e-3)

    first = calibrate_threshold(
        tmp_path / "anomaly_model.json",
        tmp_path / "preprocessing_parameters.json",
    )
    second = calibrate_threshold(
        tmp_path / "anomaly_model.json",
        tmp_path / "preprocessing_parameters.json",
    )

    assert np.isclose(first["threshold"], second["threshold"])
