"""Python training -> preprocessing -> evaluation integration test."""
from __future__ import annotations

import json

from python.evaluation.evaluate import evaluate
from python.training.train_model import train


def test_training_evaluation_deployment_metadata_chain(tmp_path):
    train(tmp_path, seed=42, epochs=35, batch_size=64, learning_rate=1.0e-3)

    result = evaluate(
        tmp_path / "anomaly_model.json",
        tmp_path / "preprocessing_parameters.json",
        tmp_path / "model_metadata.json",
    )

    model = json.loads(
        (tmp_path / "anomaly_model.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (tmp_path / "model_metadata.json").read_text(encoding="utf-8")
    )

    assert model["model_type"] == "dense_autoencoder"
    assert model["architecture"] == [80, 32, 8, 32, 80]
    assert model["parameter_count"] == 5784
    assert metadata["threshold_status"] == "calibrated"
    assert metadata["threshold"] == result["threshold"]
    assert metadata["threshold"] > 0.0

    assert set(result["scenarios"]) == {
        "normal",
        "temperature_drift",
        "vibration_fault",
        "current_fault",
        "multi_sensor_fault",
    }
