#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "../../Inc/TinyML/tinyml.h"
#include "../../Inc/TinyML/model_weights.h"

static bool initialized = false;

static float dequantize(
    int8_t value,
    float scale
)
{
    return ((float)value) * scale;
}

static void dense_relu(
    const float *input,
    uint32_t input_size,
    const int8_t *weights,
    float weight_scale,
    const int8_t *bias,
    float bias_scale,
    float *output,
    uint32_t output_size
)
{
    for (uint32_t row = 0U; row < output_size; ++row) {
        float accumulator =
            dequantize(
                bias[row],
                bias_scale
            );

        for (uint32_t column = 0U;
             column < input_size;
             ++column) {

            accumulator +=
                input[column] *
                dequantize(
                    weights[row * input_size + column],
                    weight_scale
                );
        }

        output[row] =
            accumulator > 0.0F
                ? accumulator
                : 0.0F;
    }
}

static void dense_linear(
    const float *input,
    uint32_t input_size,
    const int8_t *weights,
    float weight_scale,
    const int8_t *bias,
    float bias_scale,
    float *output,
    uint32_t output_size
)
{
    for (uint32_t row = 0U; row < output_size; ++row) {
        float accumulator =
            dequantize(
                bias[row],
                bias_scale
            );

        for (uint32_t column = 0U;
             column < input_size;
             ++column) {

            accumulator +=
                input[column] *
                dequantize(
                    weights[row * input_size + column],
                    weight_scale
                );
        }

        output[row] = accumulator;
    }
}

status_t tinyml_init(void)
{
    initialized = true;
    return STATUS_OK;
}

status_t tinyml_infer_with_contributions(
    const float input[TINYML_INPUT_SIZE],
    float *anomaly_score,
    float feature_error[TINYML_FEATURE_COUNT]
)
{
    if (input == NULL ||
        anomaly_score == NULL ||
        feature_error == NULL) {
        return STATUS_INVALID_ARG;
    }

    if (!initialized) {
        return STATUS_NOT_READY;
    }

    for (uint32_t i = 0U;
         i < TINYML_INPUT_SIZE;
         ++i) {

        if (!isfinite(input[i])) {
            return STATUS_INVALID_DATA;
        }
    }

    float encoder_output[MODEL_ENCODER_SIZE];
    float latent_output[MODEL_LATENT_SIZE];
    float decoder_output[MODEL_DECODER_SIZE];
    float reconstruction[MODEL_OUTPUT_SIZE];

    dense_relu(
        input,
        MODEL_INPUT_SIZE,
        model_encoder_weights,
        model_encoder_weight_scale,
        model_encoder_bias,
        model_encoder_bias_scale,
        encoder_output,
        MODEL_ENCODER_SIZE
    );

    dense_relu(
        encoder_output,
        MODEL_ENCODER_SIZE,
        model_latent_weights,
        model_latent_weight_scale,
        model_latent_bias,
        model_latent_bias_scale,
        latent_output,
        MODEL_LATENT_SIZE
    );

    dense_relu(
        latent_output,
        MODEL_LATENT_SIZE,
        model_decoder_weights,
        model_decoder_weight_scale,
        model_decoder_bias,
        model_decoder_bias_scale,
        decoder_output,
        MODEL_DECODER_SIZE
    );

    dense_linear(
        decoder_output,
        MODEL_DECODER_SIZE,
        model_output_weights,
        model_output_weight_scale,
        model_output_bias,
        model_output_bias_scale,
        reconstruction,
        MODEL_OUTPUT_SIZE
    );

    for (uint32_t feature = 0U;
         feature < TINYML_FEATURE_COUNT;
         ++feature) {

        feature_error[feature] = 0.0F;
    }

    float total_error = 0.0F;

    for (uint32_t sample = 0U;
         sample < TINYML_WINDOW_LENGTH;
         ++sample) {

        for (uint32_t feature = 0U;
             feature < TINYML_FEATURE_COUNT;
             ++feature) {

            const uint32_t index =
                sample * TINYML_FEATURE_COUNT + feature;

            const float difference =
                input[index] - reconstruction[index];

            const float squared_error =
                difference * difference;

            feature_error[feature] +=
                squared_error;

            total_error +=
                squared_error;
        }
    }

    for (uint32_t feature = 0U;
         feature < TINYML_FEATURE_COUNT;
         ++feature) {

        feature_error[feature] /=
            (float)TINYML_WINDOW_LENGTH;

        if (!isfinite(feature_error[feature]) ||
            feature_error[feature] < 0.0F) {
            return STATUS_ERROR;
        }
    }

    total_error /=
        (float)TINYML_INPUT_SIZE;

    if (!isfinite(total_error) ||
        total_error < 0.0F) {
        return STATUS_ERROR;
    }

    *anomaly_score = total_error;

    return STATUS_OK;
}

status_t tinyml_infer(
    const float input[TINYML_INPUT_SIZE],
    float *anomaly_score
)
{
    float feature_error[TINYML_FEATURE_COUNT];

    return tinyml_infer_with_contributions(
        input,
        anomaly_score,
        feature_error
    );
}