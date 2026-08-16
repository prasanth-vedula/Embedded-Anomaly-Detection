"""Validation tests for the dense-autoencoder INT8 deployment artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from python.quantization.quantize_model import quantize


ARCHITECTURE = [80, 32, 8, 32, 80]
PARAMETER_COUNT = 5784

LAYER_NAMES = [
    "encoder",
    "latent",
    "decoder",
    "output",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parameter_count(model: dict) -> int:
    total = 0

    for layer in LAYER_NAMES:
        total += int(np.asarray(
            model["parameters"][layer]["weight"],
            dtype=np.float32,
        ).size)

        total += int(np.asarray(
            model["parameters"][layer]["bias"],
            dtype=np.float32,
        ).size)

    return total


def _fp32_forward(model: dict, x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)

    for index, layer in enumerate(LAYER_NAMES):
        weights = np.asarray(
            model["parameters"][layer]["weight"],
            dtype=np.float32,
        )

        bias = np.asarray(
            model["parameters"][layer]["bias"],
            dtype=np.float32,
        )

        value = value @ weights.T + bias

        if index < len(LAYER_NAMES) - 1:
            value = np.maximum(value, 0.0)

    return value


def _dequantized_forward(model: dict, x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)

    for index, layer in enumerate(LAYER_NAMES):
        layer_data = model["weights"][layer]

        weights = (
            np.asarray(
                layer_data["weight_int8"],
                dtype=np.float32,
            )
            * np.float32(layer_data["weight_scale"])
        )

        bias = (
            np.asarray(
                layer_data["bias_int8"],
                dtype=np.float32,
            )
            * np.float32(layer_data["bias_scale"])
        )

        value = value @ weights.T + bias

        if index < len(LAYER_NAMES) - 1:
            value = np.maximum(value, 0.0)

    return value


def _reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    difference = original - reconstructed

    return np.mean(
        np.square(difference),
        axis=1,
        dtype=np.float32,
    )


def test_quantized_artifact_has_correct_structure(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    result = quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    assert output.exists()

    assert result["model_type"] == "dense_autoencoder_int8"
    assert result["architecture"] == ARCHITECTURE
    assert result["parameter_count"] == PARAMETER_COUNT

    assert list(result["weights"].keys()) == LAYER_NAMES

    for layer in LAYER_NAMES:
        assert "weight_int8" in result["weights"][layer]
        assert "weight_scale" in result["weights"][layer]
        assert "bias_int8" in result["weights"][layer]
        assert "bias_scale" in result["weights"][layer]


def test_quantized_tensor_shapes_and_ranges(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    model = _load_json(output)

    expected_shapes = {
        "encoder": ((32, 80), (32,)),
        "latent": ((8, 32), (8,)),
        "decoder": ((32, 8), (32,)),
        "output": ((80, 32), (80,)),
    }

    total_parameters = 0

    for layer in LAYER_NAMES:
        weights = np.asarray(
            model["weights"][layer]["weight_int8"],
            dtype=np.int16,
        )

        bias = np.asarray(
            model["weights"][layer]["bias_int8"],
            dtype=np.int16,
        )

        assert weights.shape == expected_shapes[layer][0]
        assert bias.shape == expected_shapes[layer][1]

        assert np.all(weights >= -127)
        assert np.all(weights <= 127)

        assert np.all(bias >= -127)
        assert np.all(bias <= 127)

        weight_scale = float(
            model["weights"][layer]["weight_scale"]
        )

        bias_scale = float(
            model["weights"][layer]["bias_scale"]
        )

        assert np.isfinite(weight_scale)
        assert np.isfinite(bias_scale)

        assert weight_scale > 0.0
        assert bias_scale > 0.0

        total_parameters += weights.size
        total_parameters += bias.size

    assert total_parameters == PARAMETER_COUNT


def test_quantized_scales_reconstruct_original_ranges(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    fp32_model = _load_json(
        Path("models/anomaly_model.json")
    )

    int8_model = _load_json(output)

    for layer in LAYER_NAMES:
        original_weights = np.asarray(
            fp32_model["parameters"][layer]["weight"],
            dtype=np.float32,
        )

        original_bias = np.asarray(
            fp32_model["parameters"][layer]["bias"],
            dtype=np.float32,
        )

        quantized_weights = (
            np.asarray(
                int8_model["weights"][layer]["weight_int8"],
                dtype=np.float32,
            )
            * np.float32(
                int8_model["weights"][layer]["weight_scale"]
            )
        )

        quantized_bias = (
            np.asarray(
                int8_model["weights"][layer]["bias_int8"],
                dtype=np.float32,
            )
            * np.float32(
                int8_model["weights"][layer]["bias_scale"]
            )
        )

        weight_error = np.max(
            np.abs(
                original_weights
                - quantized_weights
            )
        )

        bias_error = np.max(
            np.abs(
                original_bias
                - quantized_bias
            )
        )

        weight_scale = float(
            int8_model["weights"][layer]["weight_scale"]
        )

        bias_scale = float(
            int8_model["weights"][layer]["bias_scale"]
        )

        assert weight_error <= (
            weight_scale / 2.0 + 1.0e-6
        )

        assert bias_error <= (
            bias_scale / 2.0 + 1.0e-6
        )


def test_dequantized_inference_remains_close_to_fp32(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    fp32_model = _load_json(
        Path("models/anomaly_model.json")
    )

    int8_model = _load_json(output)

    rng = np.random.default_rng(42)

    inputs = rng.normal(
        0.0,
        1.0,
        size=(64, 80),
    ).astype(np.float32)

    fp32_output = _fp32_forward(
        fp32_model,
        inputs,
    )

    dequantized_output = _dequantized_forward(
        int8_model,
        inputs,
    )

    assert fp32_output.shape == (64, 80)
    assert dequantized_output.shape == (64, 80)

    assert np.isfinite(fp32_output).all()
    assert np.isfinite(dequantized_output).all()

    absolute_error = np.abs(
        fp32_output
        - dequantized_output
    )

    mean_absolute_error = float(
        np.mean(absolute_error)
    )

    max_absolute_error = float(
        np.max(absolute_error)
    )

    assert mean_absolute_error < 0.08
    assert max_absolute_error < 0.50


def test_anomaly_scores_remain_close_after_quantization(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    fp32_model = _load_json(
        Path("models/anomaly_model.json")
    )

    int8_model = _load_json(output)

    rng = np.random.default_rng(123)

    inputs = rng.normal(
        0.0,
        1.0,
        size=(128, 80),
    ).astype(np.float32)

    fp32_reconstruction = _fp32_forward(
        fp32_model,
        inputs,
    )

    int8_reconstruction = _dequantized_forward(
        int8_model,
        inputs,
    )

    fp32_error = _reconstruction_error(
        inputs,
        fp32_reconstruction,
    )

    int8_error = _reconstruction_error(
        inputs,
        int8_reconstruction,
    )

    score_difference = np.abs(
        fp32_error
        - int8_error
    )

    assert np.isfinite(fp32_error).all()
    assert np.isfinite(int8_error).all()

    assert float(np.mean(score_difference)) < 0.10
    assert float(np.max(score_difference)) < 0.75


def test_threshold_is_copied_from_calibrated_metadata(tmp_path):
    output = tmp_path / "anomaly_model_int8.json"

    quantize(
        input_path="models/anomaly_model.json",
        metadata_path="models/model_metadata.json",
        output_path=output,
    )

    metadata = _load_json(
        Path("models/model_metadata.json")
    )

    quantized = _load_json(output)

    assert metadata["threshold_status"] == "calibrated"

    assert (
        quantized["anomaly_detection"]["threshold"]
        == metadata["threshold"]
    )


def test_fp32_parameter_count_is_exact():
    model = _load_json(
        Path("models/anomaly_model.json")
    )

    assert model["architecture"] == ARCHITECTURE

    assert _parameter_count(model) == PARAMETER_COUNT