#ifndef TINYML_H
#define TINYML_H

#include "../Common/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TINYML_INPUT_SIZE 80U
#define TINYML_FEATURE_COUNT 5U
#define TINYML_WINDOW_LENGTH 16U

status_t tinyml_init(void);

status_t tinyml_infer(
    const float input[TINYML_INPUT_SIZE],
    float *anomaly_score
);

status_t tinyml_infer_with_contributions(
    const float input[TINYML_INPUT_SIZE],
    float *anomaly_score,
    float feature_error[TINYML_FEATURE_COUNT]
);

#ifdef __cplusplus
}
#endif

#endif