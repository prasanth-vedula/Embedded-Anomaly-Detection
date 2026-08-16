/*
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

#define MODEL_ANOMALY_THRESHOLD (0.1929460669F)

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
