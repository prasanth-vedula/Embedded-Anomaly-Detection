"""Train and serialize the compact 80-32-8-32-80 industrial anomaly autoencoder.

This module intentionally uses NumPy only so the training pipeline remains lightweight,
deterministic, reproducible, and independent of a heavyweight ML runtime. The deployed
model is a dense ReLU autoencoder with a linear reconstruction layer.

Architecture:
    80 -> 32 -> 8 -> 32 -> 80

The model is trained only on healthy windows. Anomaly threshold calibration is a
separate evaluation/deployment step and is deliberately not performed here.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from python.data.generate_dataset import generate_dataset
from python.preprocessing.pipeline import Scaler, fit_scaler, windows

INPUT_SIZE = 80
ENCODER_SIZE = 32
LATENT_SIZE = 8
DECODER_SIZE = 32
OUTPUT_SIZE = 80

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "linear"

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 250
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1.0e-3

TRAIN_SAMPLES = 1500


@dataclass
class Autoencoder:
    """Dense 80-32-8-32-80 autoencoder."""

    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    latent_weight: np.ndarray
    latent_bias: np.ndarray
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    @property
    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def parameters(self) -> Tuple[np.ndarray, ...]:
        return (
            self.encoder_weight,
            self.encoder_bias,
            self.latent_weight,
            self.latent_bias,
            self.decoder_weight,
            self.decoder_bias,
            self.output_weight,
            self.output_bias,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return reconstructed vectors for shape (N, 80) or (80,)."""
        x_array = np.asarray(x, dtype=np.float32)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
            squeeze = True
        elif x_array.ndim == 2:
            squeeze = False
        else:
            raise ValueError("input must have shape (80,) or (N, 80)")

        if x_array.shape[1] != INPUT_SIZE:
            raise ValueError(f"input must contain {INPUT_SIZE} values")

        encoder_pre = x_array @ self.encoder_weight.T + self.encoder_bias
        encoder = np.maximum(encoder_pre, 0.0)

        latent_pre = encoder @ self.latent_weight.T + self.latent_bias
        latent = np.maximum(latent_pre, 0.0)

        decoder_pre = latent @ self.decoder_weight.T + self.decoder_bias
        decoder = np.maximum(decoder_pre, 0.0)

        reconstruction = decoder @ self.output_weight.T + self.output_bias

        if squeeze:
            return reconstruction[0]
        return reconstruction

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Return mean squared reconstruction error per input vector."""
        x_array = np.asarray(x, dtype=np.float32)
        reconstruction = self.forward(x_array)
        if x_array.ndim == 1:
            return np.asarray(np.mean((x_array - reconstruction) ** 2), dtype=np.float32)
        return np.mean((x_array - reconstruction) ** 2, axis=1).astype(np.float32)


def _he_initialise(
    rng: np.random.Generator, fan_in: int, fan_out: int
) -> np.ndarray:
    """Kaiming/He normal initialisation for a ReLU dense layer."""
    scale = np.sqrt(2.0 / float(fan_in))
    return (rng.standard_normal((fan_out, fan_in)) * scale).astype(np.float32)


def _create_model(seed: int) -> Autoencoder:
    rng = np.random.default_rng(seed)
    return Autoencoder(
        encoder_weight=_he_initialise(rng, INPUT_SIZE, ENCODER_SIZE),
        encoder_bias=np.zeros(ENCODER_SIZE, dtype=np.float32),
        latent_weight=_he_initialise(rng, ENCODER_SIZE, LATENT_SIZE),
        latent_bias=np.zeros(LATENT_SIZE, dtype=np.float32),
        decoder_weight=_he_initialise(rng, LATENT_SIZE, DECODER_SIZE),
        decoder_bias=np.zeros(DECODER_SIZE, dtype=np.float32),
        output_weight=_he_initialise(rng, DECODER_SIZE, OUTPUT_SIZE),
        output_bias=np.zeros(OUTPUT_SIZE, dtype=np.float32),
    )


def _forward_cache(
    model: Autoencoder, x: np.ndarray
) -> Tuple[np.ndarray, ...]:
    z1 = x @ model.encoder_weight.T + model.encoder_bias
    a1 = np.maximum(z1, 0.0)

    z2 = a1 @ model.latent_weight.T + model.latent_bias
    a2 = np.maximum(z2, 0.0)

    z3 = a2 @ model.decoder_weight.T + model.decoder_bias
    a3 = np.maximum(z3, 0.0)

    y = a3 @ model.output_weight.T + model.output_bias
    return z1, a1, z2, a2, z3, a3, y


def _batch_gradients(
    model: Autoencoder, x: np.ndarray
) -> Tuple[np.ndarray, ...]:
    z1, a1, z2, a2, z3, a3, y = _forward_cache(model, x)
    batch_size = x.shape[0]

    dy = (2.0 / float(batch_size * OUTPUT_SIZE)) * (y - x)

    grad_output_weight = dy.T @ a3
    grad_output_bias = np.sum(dy, axis=0)

    da3 = dy @ model.output_weight
    dz3 = da3 * (z3 > 0.0)

    grad_decoder_weight = dz3.T @ a2
    grad_decoder_bias = np.sum(dz3, axis=0)

    da2 = dz3 @ model.decoder_weight
    dz2 = da2 * (z2 > 0.0)

    grad_latent_weight = dz2.T @ a1
    grad_latent_bias = np.sum(dz2, axis=0)

    da1 = dz2 @ model.latent_weight
    dz1 = da1 * (z1 > 0.0)

    grad_encoder_weight = dz1.T @ x
    grad_encoder_bias = np.sum(dz1, axis=0)

    return (
        grad_encoder_weight.astype(np.float32),
        grad_encoder_bias.astype(np.float32),
        grad_latent_weight.astype(np.float32),
        grad_latent_bias.astype(np.float32),
        grad_decoder_weight.astype(np.float32),
        grad_decoder_bias.astype(np.float32),
        grad_output_weight.astype(np.float32),
        grad_output_bias.astype(np.float32),
    )


def _adam_step(
    model: Autoencoder,
    gradients: Tuple[np.ndarray, ...],
    moments: Tuple[np.ndarray, ...],
    velocities: Tuple[np.ndarray, ...],
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> None:
    for parameter, gradient, first, second in zip(
        model.parameters(), gradients, moments, velocities
    ):
        first *= beta1
        first += (1.0 - beta1) * gradient

        second *= beta2
        second += (1.0 - beta2) * (gradient * gradient)

        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)

        parameter -= (
            learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        ).astype(np.float32)


def _mse(model: Autoencoder, x: np.ndarray) -> float:
    return float(np.mean(model.reconstruction_error(x)))


def _split_windows(
    healthy_windows: np.ndarray, validation_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    count = healthy_windows.shape[0]
    if count < 2:
        raise ValueError("at least two healthy windows are required")

    split = int(round(count * (1.0 - validation_fraction)))
    split = min(max(split, 1), count - 1)
    return healthy_windows[:split], healthy_windows[split:]


def _json_parameters(model: Autoencoder) -> Dict[str, object]:
    return {
        "encoder": {
            "weight": model.encoder_weight.tolist(),
            "bias": model.encoder_bias.tolist(),
        },
        "latent": {
            "weight": model.latent_weight.tolist(),
            "bias": model.latent_bias.tolist(),
        },
        "decoder": {
            "weight": model.decoder_weight.tolist(),
            "bias": model.decoder_bias.tolist(),
        },
        "output": {
            "weight": model.output_weight.tolist(),
            "bias": model.output_bias.tolist(),
        },
    }


def save_model(
    model: Autoencoder,
    path: pathlib.Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_loss: float,
    validation_loss: float,
) -> None:
    """Serialize the trained floating-point model."""
    payload = {
        "model_type": "dense_autoencoder",
        "version": 1,
        "architecture": [INPUT_SIZE, ENCODER_SIZE, LATENT_SIZE, DECODER_SIZE, OUTPUT_SIZE],
        "hidden_activation": HIDDEN_ACTIVATION,
        "output_activation": OUTPUT_ACTIVATION,
        "input_size": INPUT_SIZE,
        "latent_size": LATENT_SIZE,
        "parameter_count": model.parameter_count,
        "training": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        "parameters": _json_parameters(model),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(path: pathlib.Path | str) -> Autoencoder:
    """Load a floating-point autoencoder from the exported JSON artifact."""
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    if payload.get("model_type") != "dense_autoencoder":
        raise ValueError("unsupported model_type")
    if payload.get("architecture") != [
        INPUT_SIZE,
        ENCODER_SIZE,
        LATENT_SIZE,
        DECODER_SIZE,
        OUTPUT_SIZE,
    ]:
        raise ValueError("model architecture does not match 80-32-8-32-80")

    parameters = payload["parameters"]

    def array(section: str, name: str) -> np.ndarray:
        return np.asarray(parameters[section][name], dtype=np.float32)

    model = Autoencoder(
        encoder_weight=array("encoder", "weight"),
        encoder_bias=array("encoder", "bias"),
        latent_weight=array("latent", "weight"),
        latent_bias=array("latent", "bias"),
        decoder_weight=array("decoder", "weight"),
        decoder_bias=array("decoder", "bias"),
        output_weight=array("output", "weight"),
        output_bias=array("output", "bias"),
    )

    _validate_model_shapes(model)

    if int(payload.get("parameter_count", -1)) != model.parameter_count:
        raise ValueError("serialized parameter_count does not match model tensors")

    return model


def _validate_model_shapes(model: Autoencoder) -> None:
    expected = {
        "encoder_weight": (ENCODER_SIZE, INPUT_SIZE),
        "encoder_bias": (ENCODER_SIZE,),
        "latent_weight": (LATENT_SIZE, ENCODER_SIZE),
        "latent_bias": (LATENT_SIZE,),
        "decoder_weight": (DECODER_SIZE, LATENT_SIZE),
        "decoder_bias": (DECODER_SIZE,),
        "output_weight": (OUTPUT_SIZE, DECODER_SIZE),
        "output_bias": (OUTPUT_SIZE,),
    }
    for name, shape in expected.items():
        actual = getattr(model, name).shape
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, expected {shape}")
        if not np.isfinite(getattr(model, name)).all():
            raise ValueError(f"{name} contains non-finite values")


def train(
    output: str | pathlib.Path = "models",
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    validation_fraction: float = 0.20,
) -> Dict[str, object]:
    """Train on healthy data and save the model/scaler/metadata artifacts."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    out = pathlib.Path(output)
    out.mkdir(parents=True, exist_ok=True)

    healthy, _ = generate_dataset()
    healthy_training = healthy[:TRAIN_SAMPLES]

    scaler: Scaler = fit_scaler(healthy_training)
    scaled = scaler.transform(healthy_training)
    healthy_windows = windows(scaled).reshape(-1, INPUT_SIZE).astype(np.float32)

    train_x, validation_x = _split_windows(
        healthy_windows, validation_fraction
    )

    model = _create_model(seed)
    moments = tuple(np.zeros_like(p) for p in model.parameters())
    velocities = tuple(np.zeros_like(p) for p in model.parameters())

    rng = np.random.default_rng(seed + 1)
    global_step = 0

    for _epoch in range(epochs):
        order = rng.permutation(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            batch_indices = order[start : start + batch_size]
            batch = train_x[batch_indices]

            gradients = _batch_gradients(model, batch)
            global_step += 1
            _adam_step(
                model,
                gradients,
                moments,
                velocities,
                global_step,
                learning_rate,
            )

    _validate_model_shapes(model)

    train_loss = _mse(model, train_x)
    validation_loss = _mse(model, validation_x)

    scaler.save(out / "preprocessing_parameters.json")

    save_model(
        model,
        out / "anomaly_model.json",
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_loss=train_loss,
        validation_loss=validation_loss,
    )

    metadata = {
        "model_type": "dense_autoencoder",
        "architecture": [INPUT_SIZE, ENCODER_SIZE, LATENT_SIZE, DECODER_SIZE, OUTPUT_SIZE],
        "hidden_activation": HIDDEN_ACTIVATION,
        "output_activation": OUTPUT_ACTIVATION,
        "parameter_count": model.parameter_count,
        "training_samples": int(train_x.shape[0]),
        "validation_samples": int(validation_x.shape[0]),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "threshold_status": "not_calibrated",
    }
    (out / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
