"""Deployment artifact consistency tests.

These tests verify that the Python training artifacts, quantized deployment
artifact, exported embedded model metadata, preprocessing parameters, and
anomaly threshold remain internally consistent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "anomaly_model.json"
INT8_MODEL_PATH = ROOT / "models" / "anomaly_model_int8.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
PREPROCESSING_PATH = ROOT / "models" / "preprocessing_parameters.json"
WEIGHTS_HEADER_PATH = (
    ROOT / "embedded" / "Inc" / "TinyML" / "model_weights.h"
)


def load_json(path: Path) -> dict:
    assert path.exists(), f"Missing deployment artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def load_header() -> str:
    assert WEIGHTS_HEADER_PATH.exists(), (
        f"Missing exported embedded header: {WEIGHTS_HEADER_PATH}"
    )

    return WEIGHTS_HEADER_PATH.read_text(
        encoding="utf-8"
    )


def extract_define(header: str, name: str) -> str:
    """Extract a C preprocessor define without depending on whitespace."""
    pattern = rf"^\s*#define\s+{re.escape(name)}\s+(.+?)\s*$"

    match = re.search(
        pattern,
        header,
        flags=re.MULTILINE,
    )

    assert match is not None, (
        f"Missing #define {name} in exported model header"
    )

    return match.group(1)


def test_fp32_and_int8_model_architecture_match():
    fp32 = load_json(MODEL_PATH)
    int8 = load_json(INT8_MODEL_PATH)

    expected_architecture = [80, 32, 8, 32, 80]

    assert fp32["architecture"] == expected_architecture
    assert int8["architecture"] == expected_architecture

    assert fp32["parameter_count"] == 5784
    assert int8["parameter_count"] == 5784


def test_deployment_threshold_is_identical_across_artifacts():
    metadata = load_json(METADATA_PATH)
    int8 = load_json(INT8_MODEL_PATH)

    assert metadata["threshold_status"] == "calibrated"

    python_threshold = float(metadata["threshold"])
    embedded_threshold = float(
        int8["anomaly_detection"]["threshold"]
    )

    assert np.isfinite(python_threshold)
    assert python_threshold > 0.0

    assert np.isclose(
        python_threshold,
        embedded_threshold,
        rtol=0.0,
        atol=1.0e-7,
    )


def test_preprocessing_artifact_matches_deployment_dimensions():
    preprocessing = load_json(PREPROCESSING_PATH)

    assert preprocessing["features"] == 5
    assert preprocessing["window"] == 16
    assert preprocessing["input_size"] == 80

    mean = np.asarray(
        preprocessing["mean"],
        dtype=np.float32,
    )

    scale = np.asarray(
        preprocessing["scale"],
        dtype=np.float32,
    )

    assert mean.shape == (5,)
    assert scale.shape == (5,)

    assert np.isfinite(mean).all()
    assert np.isfinite(scale).all()
    assert np.all(scale > 0.0)


def test_exported_embedded_header_matches_model():
    metadata = load_json(METADATA_PATH)
    header = load_header()

    expected_defines = {
        "MODEL_INPUT_SIZE": "80U",
        "MODEL_ENCODER_SIZE": "32U",
        "MODEL_LATENT_SIZE": "8U",
        "MODEL_DECODER_SIZE": "32U",
        "MODEL_OUTPUT_SIZE": "80U",
        "MODEL_PARAMETER_COUNT": "5784U",
    }

    for name, expected_value in expected_defines.items():
        actual_value = extract_define(
            header,
            name,
        )

        assert actual_value == expected_value, (
            f"{name}: expected {expected_value}, "
            f"got {actual_value}"
        )

    threshold_value = extract_define(
        header,
        "MODEL_ANOMALY_THRESHOLD",
    )

    expected_threshold = (
        f"({float(metadata['threshold']):.10f}F)"
    )

    assert threshold_value == expected_threshold


def test_all_deployment_layers_exist():
    int8 = load_json(INT8_MODEL_PATH)

    assert list(int8["weights"].keys()) == [
        "encoder",
        "latent",
        "decoder",
        "output",
    ]

    expected_shapes = {
        "encoder": (32, 80),
        "latent": (8, 32),
        "decoder": (32, 8),
        "output": (80, 32),
    }

    for layer_name, shape in expected_shapes.items():
        layer = int8["weights"][layer_name]

        weights = np.asarray(
            layer["weight_int8"],
            dtype=np.int8,
        )

        biases = np.asarray(
            layer["bias_int8"],
            dtype=np.int8,
        )

        assert weights.shape == shape
        assert biases.shape == (shape[0],)

        assert np.isfinite(
            weights.astype(np.float32)
        ).all()

        assert np.isfinite(
            biases.astype(np.float32)
        ).all()