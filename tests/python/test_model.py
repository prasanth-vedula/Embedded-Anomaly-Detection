"""Tests for the floating-point compact autoencoder."""
from __future__ import annotations

import json
import numpy as np

from python.training.train_model import (
    DECODER_SIZE,
    ENCODER_SIZE,
    INPUT_SIZE,
    LATENT_SIZE,
    OUTPUT_SIZE,
    Autoencoder,
    load_model,
    train,
)


def test_architecture_and_parameter_count():
    assert [INPUT_SIZE, ENCODER_SIZE, LATENT_SIZE, DECODER_SIZE, OUTPUT_SIZE] == [
        80,
        32,
        8,
        32,
        80,
    ]

    rng = np.random.default_rng(1)
    model = Autoencoder(
        rng.normal(size=(32, 80)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        rng.normal(size=(8, 32)).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        rng.normal(size=(32, 8)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        rng.normal(size=(80, 32)).astype(np.float32),
        np.zeros(80, dtype=np.float32),
    )
    assert model.parameter_count == 5784


def test_forward_shape_and_nonnegative_reconstruction_error():
    rng = np.random.default_rng(2)
    model = Autoencoder(
        rng.normal(size=(32, 80)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        rng.normal(size=(8, 32)).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        rng.normal(size=(32, 8)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        rng.normal(size=(80, 32)).astype(np.float32),
        np.zeros(80, dtype=np.float32),
    )
    x = rng.normal(size=(7, 80)).astype(np.float32)

    reconstruction = model.forward(x)
    errors = model.reconstruction_error(x)

    assert reconstruction.shape == (7, 80)
    assert errors.shape == (7,)
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(errors).all()
    assert np.all(errors >= 0.0)


def test_training_produces_real_autoencoder(tmp_path):
    metadata = train(
        tmp_path,
        seed=123,
        epochs=40,
        batch_size=64,
        learning_rate=1.0e-3,
    )

    assert metadata["model_type"] == "dense_autoencoder"
    assert metadata["architecture"] == [80, 32, 8, 32, 80]
    assert metadata["parameter_count"] == 5784
    assert metadata["threshold_status"] == "not_calibrated"
    assert metadata["train_loss"] >= 0.0
    assert metadata["validation_loss"] >= 0.0

    model_path = tmp_path / "anomaly_model.json"
    scaler_path = tmp_path / "preprocessing_parameters.json"
    metadata_path = tmp_path / "model_metadata.json"

    assert model_path.exists()
    assert scaler_path.exists()
    assert metadata_path.exists()

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["model_type"] == "dense_autoencoder"
    assert payload["architecture"] == [80, 32, 8, 32, 80]
    assert payload["parameter_count"] == 5784

    model = load_model(model_path)
    rng = np.random.default_rng(99)
    x = rng.normal(size=(4, 80)).astype(np.float32)

    reconstructed = model.forward(x)
    errors = model.reconstruction_error(x)

    assert reconstructed.shape == (4, 80)
    assert errors.shape == (4,)
    assert np.isfinite(reconstructed).all()
    assert np.isfinite(errors).all()


def test_saved_model_reload_is_numerically_consistent(tmp_path):
    train(
        tmp_path,
        seed=456,
        epochs=20,
        batch_size=64,
        learning_rate=1.0e-3,
    )

    model = load_model(tmp_path / "anomaly_model.json")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 80)).astype(np.float32)

    first = model.forward(x)
    second = load_model(tmp_path / "anomaly_model.json").forward(x)

    assert np.array_equal(first, second)
