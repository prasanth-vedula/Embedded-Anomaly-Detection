/*
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
