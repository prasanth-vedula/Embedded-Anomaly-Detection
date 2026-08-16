"""Deterministic INT8 export for the dense autoencoder."""

from __future__ import annotations

import json
import pathlib
import numpy as np


ARCHITECTURE = [80, 32, 8, 32, 80]
PARAMETER_COUNT = 5784


def quantize_tensor(values):
    arr = np.asarray(values, dtype=np.float32)

    max_abs = float(np.max(np.abs(arr)))

    if max_abs < 1.0e-12:
        scale = 1.0
        q = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / 127.0
        q = np.rint(arr / scale)
        q = np.clip(q, -127, 127).astype(np.int8)

    return q, scale


def quantize(
    input_path="models/anomaly_model.json",
    metadata_path="models/model_metadata.json",
    output_path="models/anomaly_model_int8.json",
):

    input_path = pathlib.Path(input_path)
    metadata_path = pathlib.Path(metadata_path)
    output_path = pathlib.Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # ---------------------------------------------------------
    # Validate FP32 model
    # ---------------------------------------------------------

    if model.get("model_type") != "dense_autoencoder":
        raise ValueError("Expected dense_autoencoder model.")

    if model.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"Unexpected architecture: {model.get('architecture')}"
        )

    if model.get("parameter_count") != PARAMETER_COUNT:
        raise ValueError(
            f"Unexpected parameter count: "
            f"{model.get('parameter_count')}"
        )

    parameters = model["parameters"]

    required_layers = ["encoder", "latent", "decoder", "output"]

    for layer in required_layers:
        if layer not in parameters:
            raise ValueError(f"Missing layer: {layer}")

        if "weight" not in parameters[layer]:
            raise ValueError(f"Missing weight in layer: {layer}")

        if "bias" not in parameters[layer]:
            raise ValueError(f"Missing bias in layer: {layer}")

    # ---------------------------------------------------------
    # Quantize every layer
    # ---------------------------------------------------------

    quantized = {}

    for layer in required_layers:

        weight_q, weight_scale = quantize_tensor(
            parameters[layer]["weight"]
        )

        bias_q, bias_scale = quantize_tensor(
            parameters[layer]["bias"]
        )

        quantized[layer] = {
            "weight_int8": weight_q.tolist(),
            "weight_scale": weight_scale,
            "bias_int8": bias_q.tolist(),
            "bias_scale": bias_scale,
        }

    # ---------------------------------------------------------
    # Threshold comes from calibrated metadata
    # ---------------------------------------------------------

    if metadata.get("threshold_status") != "calibrated":
        raise ValueError("Model threshold is not calibrated.")

    threshold = float(metadata["threshold"])

    # ---------------------------------------------------------
    # Final INT8 deployment artifact
    # ---------------------------------------------------------

    result = {
        "model_type": "dense_autoencoder_int8",
        "version": 1,

        "architecture": ARCHITECTURE,
        "input_size": 80,
        "latent_size": 8,
        "parameter_count": PARAMETER_COUNT,

        "activation": {
            "hidden": "relu",
            "output": "linear",
        },

        "weights": quantized,

        "preprocessing": {
            "source": "models/preprocessing_parameters.json"
        },

        "anomaly_detection": {
            "threshold": threshold,
            "threshold_method": metadata["threshold_method"],
            "threshold_quantile": metadata["threshold_quantile"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":

    result = quantize()

    print("INT8 quantization successful")
    print(f"Model type: {result['model_type']}")
    print(f"Architecture: {result['architecture']}")
    print(f"Parameters: {result['parameter_count']}")
    print(
        f"Threshold: "
        f"{result['anomaly_detection']['threshold']}"
    )
    print("Layers: encoder, latent, decoder, output")
    print("Output: models/anomaly_model_int8.json")