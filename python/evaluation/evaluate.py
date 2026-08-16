"""Evaluate the compact autoencoder and calibrate its deployment threshold."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from python.data.generate_dataset import generate_dataset
from python.preprocessing.pipeline import (
    Scaler,
    flatten_windows,
    windows,
)
from python.training.train_model import INPUT_SIZE, TRAIN_SAMPLES, load_model

DEFAULT_MODEL_PATH = Path("models/anomaly_model.json")
DEFAULT_SCALER_PATH = Path("models/preprocessing_parameters.json")
DEFAULT_METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_THRESHOLD_QUANTILE = 0.99
DEFAULT_VALIDATION_FRACTION = 0.20


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"required artifact does not exist: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _healthy_validation_windows(
    scaler: Scaler,
    validation_fraction: float,
) -> np.ndarray:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    healthy, _ = generate_dataset()
    healthy_training = healthy[:TRAIN_SAMPLES]

    scaled = scaler.transform(healthy_training)
    all_windows = flatten_windows(windows(scaled))

    if all_windows.shape[0] < 2:
        raise ValueError("at least two healthy validation windows are required")

    split = int(round(all_windows.shape[0] * (1.0 - validation_fraction)))
    split = min(max(split, 1), all_windows.shape[0] - 1)
    return all_windows[split:]


def calibrate_threshold(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    *,
    quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> dict[str, float]:
    """Calibrate threshold exclusively from healthy validation reconstruction errors."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")

    model = load_model(model_path)
    scaler = Scaler.load(scaler_path)

    validation = _healthy_validation_windows(scaler, validation_fraction)
    errors = np.asarray(
        model.reconstruction_error(validation),
        dtype=np.float64,
    )

    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("healthy validation produced invalid reconstruction errors")

    threshold = float(np.quantile(errors, quantile))
    threshold = max(threshold, float(np.finfo(np.float32).eps))

    return {
        "threshold": threshold,
        "quantile": float(quantile),
        "validation_count": float(errors.size),
        "normal_validation_mean_error": float(np.mean(errors)),
        "normal_validation_std_error": float(np.std(errors)),
        "normal_validation_p95_error": float(np.quantile(errors, 0.95)),
        "normal_validation_p99_error": float(np.quantile(errors, 0.99)),
        "normal_validation_max_error": float(np.max(errors)),
    }


def _score_stream(
    model_path: str | Path,
    scaler_path: str | Path,
    values: np.ndarray,
) -> np.ndarray:
    model = load_model(model_path)
    scaler = Scaler.load(scaler_path)

    scaled = scaler.transform(values)
    flattened = flatten_windows(windows(scaled))

    if flattened.shape[0] == 0:
        raise ValueError("scenario produced no complete windows")

    errors = np.asarray(
        model.reconstruction_error(flattened),
        dtype=np.float32,
    )

    if errors.shape != (flattened.shape[0],):
        raise RuntimeError("model returned an invalid reconstruction-error shape")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise RuntimeError("model returned invalid reconstruction errors")

    return errors


def _scenario_metrics(
    errors: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    if errors.size == 0:
        raise ValueError("scenario contains no scores")

    return {
        "window_count": float(errors.size),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "p95_error": float(np.quantile(errors, 0.95)),
        "p99_error": float(np.quantile(errors, 0.99)),
        "max_error": float(np.max(errors)),
        "anomaly_rate": float(np.mean(errors >= threshold)),
    }


def _write_metadata(
    metadata_path: Path,
    model_payload: dict[str, Any],
    calibration: dict[str, float],
    evaluation: dict[str, dict[str, float]],
) -> None:
    existing: dict[str, Any] = {}
    if metadata_path.exists():
        existing = _load_json(metadata_path)

    existing.update(
        {
            "model_type": model_payload["model_type"],
            "architecture": model_payload["architecture"],
            "hidden_activation": model_payload["hidden_activation"],
            "output_activation": model_payload["output_activation"],
            "parameter_count": int(model_payload["parameter_count"]),
            "threshold_status": "calibrated",
            "threshold": calibration["threshold"],
            "threshold_method": "healthy_validation_quantile",
            "threshold_quantile": calibration["quantile"],
            "normal_validation_mean_error": calibration[
                "normal_validation_mean_error"
            ],
            "normal_validation_std_error": calibration[
                "normal_validation_std_error"
            ],
            "normal_validation_p95_error": calibration[
                "normal_validation_p95_error"
            ],
            "normal_validation_p99_error": calibration[
                "normal_validation_p99_error"
            ],
            "normal_validation_max_error": calibration[
                "normal_validation_max_error"
            ],
            "normal_validation_count": int(calibration["validation_count"]),
            "evaluation": evaluation,
        }
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(existing, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    *,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> dict[str, Any]:
    """Calibrate, evaluate all scenarios, and persist deployment metadata."""
    model_payload = _load_json(model_path)

    if model_payload.get("model_type") != "dense_autoencoder":
        raise ValueError("evaluation requires a dense_autoencoder model")
    if model_payload.get("architecture") != [80, 32, 8, 32, 80]:
        raise ValueError("evaluation requires architecture 80-32-8-32-80")
    if model_payload.get("parameter_count") != 5784:
        raise ValueError("evaluation requires the 5,784-parameter model")

    calibration = calibrate_threshold(
        model_path,
        scaler_path,
        quantile=threshold_quantile,
    )
    threshold = calibration["threshold"]

    healthy, faults = generate_dataset()
    expected_faults = {
        "temperature_drift",
        "vibration_fault",
        "current_fault",
        "multi_sensor_fault",
    }

    if set(faults) != expected_faults:
        raise RuntimeError(
            f"fault scenario set mismatch: expected {sorted(expected_faults)}, "
            f"received {sorted(faults)}"
        )

    scenarios = {"normal": healthy, **faults}

    evaluation = {
        name: _scenario_metrics(
            _score_stream(model_path, scaler_path, values),
            threshold,
        )
        for name, values in scenarios.items()
    }

    result: dict[str, Any] = {
        "model_type": model_payload["model_type"],
        "architecture": model_payload["architecture"],
        "parameter_count": int(model_payload["parameter_count"]),
        "threshold": threshold,
        "threshold_method": "healthy_validation_quantile",
        "threshold_quantile": float(threshold_quantile),
        "calibration": calibration,
        "scenarios": evaluation,
    }

    _write_metadata(
        Path(metadata_path),
        model_payload,
        calibration,
        evaluation,
    )

    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
