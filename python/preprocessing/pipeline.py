"""Canonical preprocessing shared by training, evaluation and deployment export."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

FEATURES = 5
WINDOW = 16
MODEL_INPUT_SIZE = FEATURES * WINDOW


@dataclass(frozen=True)
class Scaler:
    """Per-sensor affine normalization parameters."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)

        if mean.shape != (FEATURES,):
            raise ValueError(f"mean must have shape ({FEATURES},)")
        if scale.shape != (FEATURES,):
            raise ValueError(f"scale must have shape ({FEATURES},)")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("scaler parameters must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("scaler scale values must be positive")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize using float64 arithmetic before producing float32 deployment data.

        The exported parameters remain float32 so the embedded representation can
        use the same values. Float64 arithmetic here prevents avoidable cancellation
        error when a float32 mean is subtracted from sensor values.
        """
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != FEATURES:
            raise ValueError("input must have shape (samples, 5)")

        transformed = (
            values.astype(np.float64)
            - self.mean.astype(np.float64)
        ) / self.scale.astype(np.float64)

        result = transformed.astype(np.float32)

        if not np.isfinite(result).all():
            raise ValueError("preprocessing produced non-finite values")

        return result

    def inverse(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != FEATURES:
            raise ValueError("input must have shape (samples, 5)")

        restored = (
            values.astype(np.float64) * self.scale.astype(np.float64)
            + self.mean.astype(np.float64)
        )
        result = restored.astype(np.float32)

        if not np.isfinite(result).all():
            raise ValueError("inverse preprocessing produced non-finite values")

        return result

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "window": WINDOW,
            "features": FEATURES,
            "input_size": MODEL_INPUT_SIZE,
            "method": "per_sensor_standardization",
            "arithmetic": "float64_fit_and_transform_then_float32_output",
        }
        destination.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> "Scaler":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"preprocessing artifact not found: {source}"
            )

        payload = json.loads(source.read_text(encoding="utf-8"))

        if payload.get("window", WINDOW) != WINDOW:
            raise ValueError(
                "preprocessing window does not match deployment window"
            )
        if payload.get("features", FEATURES) != FEATURES:
            raise ValueError(
                "preprocessing feature count does not match deployment"
            )
        if payload.get("input_size", MODEL_INPUT_SIZE) != MODEL_INPUT_SIZE:
            raise ValueError(
                "preprocessing input size does not match deployment model"
            )

        return Scaler(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["scale"], dtype=np.float32),
        )


def fit_scaler(x: np.ndarray) -> Scaler:
    """Fit the canonical scaler on healthy training data only."""
    values = np.asarray(x, dtype=np.float32)

    if not validate(values):
        raise ValueError("cannot fit scaler on invalid sensor data")

    # Fit statistics in float64 so the exported float32 parameters are based on
    # numerically stable statistics rather than accumulated float32 rounding.
    values64 = values.astype(np.float64)
    mean = values64.mean(axis=0)
    scale = np.maximum(values64.std(axis=0), 1.0e-6)

    return Scaler(
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def windows(
    x: np.ndarray,
    window: int = WINDOW,
) -> np.ndarray:
    """Convert a sensor stream into overlapping [window, feature] windows."""
    values = np.asarray(x, dtype=np.float32)

    if window <= 0:
        raise ValueError("window must be positive")

    if values.ndim != 2 or values.shape[1] != FEATURES:
        return np.empty((0, window, FEATURES), dtype=np.float32)

    if values.shape[0] < window:
        return np.empty((0, window, FEATURES), dtype=np.float32)

    return np.stack(
        [
            values[i : i + window]
            for i in range(values.shape[0] - window + 1)
        ],
        axis=0,
    ).astype(np.float32)


def validate(x: np.ndarray) -> bool:
    """Return True only for finite two-dimensional five-sensor streams."""
    values = np.asarray(x, dtype=np.float32)

    return bool(
        values.ndim == 2
        and values.shape[1] == FEATURES
        and values.shape[0] > 0
        and np.isfinite(values).all()
    )


def flatten_windows(x: np.ndarray) -> np.ndarray:
    """Flatten [N, 16, 5] windows into the model's [N, 80] input."""
    values = np.asarray(x, dtype=np.float32)

    if values.ndim != 3 or values.shape[1:] != (WINDOW, FEATURES):
        raise ValueError("expected windows with shape (N, 16, 5)")

    return values.reshape(
        values.shape[0],
        MODEL_INPUT_SIZE,
    ).astype(np.float32)
