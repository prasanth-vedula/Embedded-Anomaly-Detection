"""Export the calibrated TinyML model and deployment parameters to C."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model_int8.json"
PREPROCESSING_PATH = (
    PROJECT_ROOT / "models" / "preprocessing_parameters.json"
)
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

TINYML_HEADER = (
    PROJECT_ROOT
    / "embedded"
    / "Inc"
    / "TinyML"
    / "model_weights.h"
)

TINYML_SOURCE = (
    PROJECT_ROOT
    / "embedded"
    / "Src"
    / "TinyML"
    / "model_weights.c"
)

DEPLOYMENT_HEADER = (
    PROJECT_ROOT
    / "embedded"
    / "Inc"
    / "Config"
    / "deployment_parameters.h"
)

DEPLOYMENT_SOURCE = (
    PROJECT_ROOT
    / "embedded"
    / "Src"
    / "Config"
    / "deployment_parameters.c"
)

EXPECTED_ARCHITECTURE = [80, 32, 8, 32, 80]
EXPECTED_PARAMETER_COUNT = 5784
EXPECTED_FEATURES = 5
EXPECTED_WINDOW = 16


LAYERS = (
    ("encoder", 32, 80),
    ("latent", 8, 32),
    ("decoder", 32, 8),
    ("output", 80, 32),
)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_model(model: dict) -> None:
    if model.get("model_type") != "dense_autoencoder_int8":
        raise ValueError("Expected dense_autoencoder_int8 model.")

    if model.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected model architecture.")

    if model.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Unexpected model parameter count.")

    if "weights" not in model:
        raise ValueError("Model has no weights.")

    for name, rows, cols in LAYERS:
        if name not in model["weights"]:
            raise ValueError(f"Missing model layer: {name}")

        layer = model["weights"][name]

        weights = layer["weight_int8"]
        bias = layer["bias_int8"]

        if len(weights) != rows:
            raise ValueError(f"{name}: invalid weight rows.")

        if any(len(row) != cols for row in weights):
            raise ValueError(f"{name}: invalid weight columns.")

        if len(bias) != rows:
            raise ValueError(f"{name}: invalid bias length.")

        if float(layer["weight_scale"]) <= 0.0:
            raise ValueError(f"{name}: invalid weight scale.")

        if float(layer["bias_scale"]) <= 0.0:
            raise ValueError(f"{name}: invalid bias scale.")


def validate_preprocessing(preprocessing: dict) -> None:
    if preprocessing.get("features") != EXPECTED_FEATURES:
        raise ValueError("Preprocessing feature count mismatch.")

    if preprocessing.get("window") != EXPECTED_WINDOW:
        raise ValueError("Preprocessing window mismatch.")

    if preprocessing.get("input_size") != (
        EXPECTED_FEATURES * EXPECTED_WINDOW
    ):
        raise ValueError("Preprocessing input size mismatch.")

    mean = preprocessing.get("mean")
    scale = preprocessing.get("scale")

    if not isinstance(mean, list) or len(mean) != EXPECTED_FEATURES:
        raise ValueError("Preprocessing mean must contain five values.")

    if not isinstance(scale, list) or len(scale) != EXPECTED_FEATURES:
        raise ValueError("Preprocessing scale must contain five values.")

    for value in mean:
        if not isinstance(value, (int, float)):
            raise ValueError("Preprocessing mean contains invalid data.")

    for value in scale:
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            raise ValueError("Preprocessing scale contains invalid data.")


def calibrated_threshold(metadata: dict) -> float:
    if metadata.get("threshold_status") != "calibrated":
        raise ValueError("Model threshold is not calibrated.")

    threshold = float(metadata["threshold"])

    if threshold <= 0.0:
        raise ValueError("Calibrated threshold must be positive.")

    return threshold


def flatten(matrix: list[list[int]]) -> list[int]:
    return [
        int(value)
        for row in matrix
        for value in row
    ]


def format_array(values: list[int], columns: int = 16) -> str:
    lines = []

    for start in range(0, len(values), columns):
        chunk = values[start:start + columns]

        lines.append(
            "    "
            + ", ".join(f"{int(value):4d}" for value in chunk)
            + ","
        )

    return "\n".join(lines)


def generate_model_header(threshold: float) -> str:
    return f"""/*
 * AUTO-GENERATED FILE.
 *
 * Source:
 *   models/anomaly_model_int8.json
 *
 * Do not edit manually.
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

#define MODEL_INPUT_SIZE       80U
#define MODEL_ENCODER_SIZE     32U
#define MODEL_LATENT_SIZE       8U
#define MODEL_DECODER_SIZE     32U
#define MODEL_OUTPUT_SIZE      80U

#define MODEL_PARAMETER_COUNT 5784U

#define MODEL_ANOMALY_THRESHOLD ({threshold:.10f}F)

extern const int8_t model_encoder_weights[32U * 80U];
extern const int8_t model_encoder_bias[32U];
extern const float model_encoder_weight_scale;
extern const float model_encoder_bias_scale;

extern const int8_t model_latent_weights[8U * 32U];
extern const int8_t model_latent_bias[8U];
extern const float model_latent_weight_scale;
extern const float model_latent_bias_scale;

extern const int8_t model_decoder_weights[32U * 8U];
extern const int8_t model_decoder_bias[32U];
extern const float model_decoder_weight_scale;
extern const float model_decoder_bias_scale;

extern const int8_t model_output_weights[80U * 32U];
extern const int8_t model_output_bias[80U];
extern const float model_output_weight_scale;
extern const float model_output_bias_scale;

#endif
"""


def generate_model_source(model: dict) -> str:
    encoder = model["weights"]["encoder"]
    latent = model["weights"]["latent"]
    decoder = model["weights"]["decoder"]
    output = model["weights"]["output"]

    return f"""/*
 * AUTO-GENERATED FILE.
 *
 * Source:
 *   models/anomaly_model_int8.json
 *
 * Do not edit manually.
 */

#include "../../Inc/TinyML/model_weights.h"

const int8_t model_encoder_weights[32U * 80U] = {{
{format_array(flatten(encoder["weight_int8"]))}
}};

const int8_t model_encoder_bias[32U] = {{
{format_array(encoder["bias_int8"])}
}};

const float model_encoder_weight_scale =
    {float(encoder["weight_scale"]):.12e}F;

const float model_encoder_bias_scale =
    {float(encoder["bias_scale"]):.12e}F;


const int8_t model_latent_weights[8U * 32U] = {{
{format_array(flatten(latent["weight_int8"]))}
}};

const int8_t model_latent_bias[8U] = {{
{format_array(latent["bias_int8"])}
}};

const float model_latent_weight_scale =
    {float(latent["weight_scale"]):.12e}F;

const float model_latent_bias_scale =
    {float(latent["bias_scale"]):.12e}F;


const int8_t model_decoder_weights[32U * 8U] = {{
{format_array(flatten(decoder["weight_int8"]))}
}};

const int8_t model_decoder_bias[32U] = {{
{format_array(decoder["bias_int8"])}
}};

const float model_decoder_weight_scale =
    {float(decoder["weight_scale"]):.12e}F;

const float model_decoder_bias_scale =
    {float(decoder["bias_scale"]):.12e}F;


const int8_t model_output_weights[80U * 32U] = {{
{format_array(flatten(output["weight_int8"]))}
}};

const int8_t model_output_bias[80U] = {{
{format_array(output["bias_int8"])}
}};

const float model_output_weight_scale =
    {float(output["weight_scale"]):.12e}F;

const float model_output_bias_scale =
    {float(output["bias_scale"]):.12e}F;
"""


def generate_deployment_header() -> str:
    return """/*
 * AUTO-GENERATED FILE.
 *
 * Sources:
 *   models/preprocessing_parameters.json
 *   models/model_metadata.json
 *
 * Do not edit manually.
 */

#ifndef DEPLOYMENT_PARAMETERS_H
#define DEPLOYMENT_PARAMETERS_H

#include <stdint.h>

#define DEPLOYMENT_FEATURE_COUNT 5U
#define DEPLOYMENT_WINDOW_LENGTH 16U
#define DEPLOYMENT_INPUT_SIZE 80U

extern const float deployment_sensor_mean[DEPLOYMENT_FEATURE_COUNT];

extern const float deployment_sensor_scale[DEPLOYMENT_FEATURE_COUNT];

extern const float deployment_anomaly_threshold;

#endif
"""


def generate_deployment_source(
    preprocessing: dict,
    threshold: float,
) -> str:
    mean = preprocessing["mean"]
    scale = preprocessing["scale"]

    mean_values = ", ".join(
        f"{float(value):.12e}F"
        for value in mean
    )

    scale_values = ", ".join(
        f"{float(value):.12e}F"
        for value in scale
    )

    return f"""/*
 * AUTO-GENERATED FILE.
 *
 * Sources:
 *   models/preprocessing_parameters.json
 *   models/model_metadata.json
 *
 * Do not edit manually.
 */

#include "../../Inc/Config/deployment_parameters.h"

const float deployment_sensor_mean[
    DEPLOYMENT_FEATURE_COUNT
] = {{
    {mean_values}
}};

const float deployment_sensor_scale[
    DEPLOYMENT_FEATURE_COUNT
] = {{
    {scale_values}
}};

const float deployment_anomaly_threshold =
    {threshold:.12e}F;
"""


def export_model() -> None:
    model = read_json(MODEL_PATH)
    preprocessing = read_json(PREPROCESSING_PATH)
    metadata = read_json(METADATA_PATH)

    validate_model(model)
    validate_preprocessing(preprocessing)

    threshold = calibrated_threshold(metadata)

    model_anomaly_threshold = float(
        model["anomaly_detection"]["threshold"]
    )

    if abs(model_anomaly_threshold - threshold) > 1.0e-7:
        raise ValueError(
            "Model threshold and metadata threshold disagree."
        )

    TINYML_HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TINYML_SOURCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEPLOYMENT_HEADER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEPLOYMENT_SOURCE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TINYML_HEADER.write_text(
        generate_model_header(threshold),
        encoding="utf-8",
    )

    TINYML_SOURCE.write_text(
        generate_model_source(model),
        encoding="utf-8",
    )

    DEPLOYMENT_HEADER.write_text(
        generate_deployment_header(),
        encoding="utf-8",
    )

    DEPLOYMENT_SOURCE.write_text(
        generate_deployment_source(
            preprocessing,
            threshold,
        ),
        encoding="utf-8",
    )

    print("Deployment export successful")
    print("Architecture: 80 -> 32 -> 8 -> 32 -> 80")
    print("Parameters: 5784")
    print(f"Threshold: {threshold:.10f}")
    print("Generated:")
    print(f"  {TINYML_HEADER}")
    print(f"  {TINYML_SOURCE}")
    print(f"  {DEPLOYMENT_HEADER}")
    print(f"  {DEPLOYMENT_SOURCE}")


if __name__ == "__main__":
    export_model()