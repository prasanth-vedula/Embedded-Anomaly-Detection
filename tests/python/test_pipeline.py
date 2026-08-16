"""Tests for the canonical preprocessing pipeline."""
from __future__ import annotations

import numpy as np

from python.data.generate_dataset import generate_dataset
from python.preprocessing.pipeline import (
    FEATURES,
    MODEL_INPUT_SIZE,
    WINDOW,
    Scaler,
    fit_scaler,
    flatten_windows,
    validate,
    windows,
)


def test_dataset_reproducible_and_contains_all_faults():
    first, first_faults = generate_dataset(seed=3)
    second, second_faults = generate_dataset(seed=3)

    assert np.array_equal(first, second)

    expected_faults = {
        "temperature_drift",
        "vibration_fault",
        "current_fault",
        "multi_sensor_fault",
    }

    assert set(first_faults) == expected_faults
    assert set(second_faults) == expected_faults

    for name in expected_faults:
        assert np.array_equal(first_faults[name], second_faults[name])
        assert first_faults[name].shape == first.shape


def test_windows_and_scaling():
    healthy, _ = generate_dataset(40)

    scaler = fit_scaler(healthy)
    scaled = scaler.transform(healthy)
    result = windows(scaled)

    assert validate(healthy)

    assert result.shape == (25, WINDOW, FEATURES)

    flattened = flatten_windows(result)

    assert flattened.shape == (
        25,
        MODEL_INPUT_SIZE,
    )

    # The deployment contract is defined by the exported float32 scaler
    # parameters. Verify that transform() uses those exact parameters rather
    # than requiring the transformed float32 array to have an exactly-zero
    # arithmetic mean.
    expected_scaled = (
        healthy.astype(np.float64)
        - scaler.mean.astype(np.float64)
    ) / scaler.scale.astype(np.float64)

    expected_scaled = expected_scaled.astype(np.float32)

    assert np.array_equal(
        scaled,
        expected_scaled,
    )

    assert np.isfinite(scaled).all()
    assert np.isfinite(flattened).all()


def test_scaler_round_trip():
    healthy, _ = generate_dataset(100)

    scaler = fit_scaler(healthy)

    transformed = scaler.transform(healthy)
    restored = scaler.inverse(transformed)

    assert np.allclose(
        restored,
        healthy,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_scaler_serialization(tmp_path):
    healthy, _ = generate_dataset(100)

    scaler = fit_scaler(healthy)

    path = tmp_path / "preprocessing_parameters.json"

    scaler.save(path)

    loaded = Scaler.load(path)

    assert np.array_equal(
        loaded.mean,
        scaler.mean,
    )

    assert np.array_equal(
        loaded.scale,
        scaler.scale,
    )


def test_invalid_data_is_rejected():
    invalid = np.zeros(
        (10, FEATURES),
        dtype=np.float32,
    )

    invalid[0, 0] = np.nan

    assert not validate(invalid)

    valid = np.zeros(
        (10, FEATURES),
        dtype=np.float32,
    )

    assert validate(valid)