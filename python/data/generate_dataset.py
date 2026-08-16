"""Deterministic industrial sensor data generation for training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

SENSOR_NAMES = ("temperature", "vibration", "pressure", "current", "humidity")
FEATURE_COUNT = len(SENSOR_NAMES)
DEFAULT_SAMPLES = 2400
DEFAULT_SEED = 7
FAULT_START = 1500


def generate_dataset(
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return a deterministic healthy stream and four independent fault streams.

    The healthy stream is used for training/calibration. Fault streams preserve
    the same baseline data and inject a physically interpretable disturbance
    from FAULT_START onward.
    """
    if samples < 32:
        raise ValueError("samples must be at least 32")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float32)
    phase = 2.0 * np.pi * t / 180.0

    temperature = (
        68.0
        + 2.0 * np.sin(phase)
        + 0.35 * np.sin(t / 17.0)
        + rng.normal(0.0, 0.25, samples)
    )
    vibration = (
        0.42
        + 0.06 * np.sin(phase * 1.7)
        + rng.normal(0.0, 0.025, samples)
    )
    pressure = (
        4.8
        + 0.18 * np.sin(phase + 0.4)
        + rng.normal(0.0, 0.035, samples)
    )
    current = (
        8.2
        + 0.55 * np.sin(phase - 0.2)
        + 0.25 * np.sin(t / 31.0)
        + rng.normal(0.0, 0.12, samples)
    )
    humidity = 45.0 + 3.0 * np.sin(phase / 2.0) + rng.normal(
        0.0, 0.5, samples
    )

    healthy = np.column_stack(
        [temperature, vibration, pressure, current, humidity]
    ).astype(np.float32)

    start = min(FAULT_START, max(0, samples // 2))
    fault_streams: Dict[str, np.ndarray] = {}

    temperature_drift = healthy.copy()
    temperature_drift[start:, 0] += np.linspace(
        0.0, 12.0, samples - start, dtype=np.float32
    )
    fault_streams["temperature_drift"] = temperature_drift

    vibration_fault = healthy.copy()
    vibration_fault[start:, 1] += (
        0.55 + 0.18 * np.sin(t[start:] / 4.0)
    ).astype(np.float32)
    fault_streams["vibration_fault"] = vibration_fault

    current_fault = healthy.copy()
    current_fault[start::29, 3] += 3.8
    fault_streams["current_fault"] = current_fault

    multi_sensor_fault = healthy.copy()
    multi_sensor_fault[start:, 0] += 7.0
    multi_sensor_fault[start:, 1] += 0.45
    multi_sensor_fault[start:, 2] += 0.7
    fault_streams["multi_sensor_fault"] = multi_sensor_fault

    if healthy.shape != (samples, FEATURE_COUNT):
        raise RuntimeError("generated healthy data has an invalid shape")
    if set(fault_streams) != {
        "temperature_drift",
        "vibration_fault",
        "current_fault",
        "multi_sensor_fault",
    }:
        raise RuntimeError("fault scenario set is incomplete")

    return healthy, fault_streams


def save_csv(path: str | Path, values: np.ndarray) -> None:
    """Save a sensor stream with the canonical sensor column names."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise ValueError("values must have shape (samples, 5)")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        destination,
        values,
        delimiter=",",
        header=",".join(SENSOR_NAMES),
        comments="",
    )


if __name__ == "__main__":
    healthy, faults = generate_dataset()
    print(f"healthy shape: {healthy.shape}")
    print(f"fault scenarios: {', '.join(sorted(faults))}")
