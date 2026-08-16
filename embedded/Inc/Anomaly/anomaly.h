#ifndef ANOMALY_H
#define ANOMALY_H

#include "../Common/types.h"

#define ANOMALY_FEATURE_COUNT 5U

typedef enum {
    ANOMALY_NORMAL = 0,
    ANOMALY_SUSPECT = 1,
    ANOMALY_CONFIRMED = 2
} anomaly_state_t;

typedef struct {
    float score;
    float feature_error[ANOMALY_FEATURE_COUNT];
    uint32_t fault_mask;
    anomaly_state_t state;
    uint32_t persistence;
} anomaly_result_t;

void anomaly_init(void);

status_t anomaly_update(
    float score,
    const float feature_error[ANOMALY_FEATURE_COUNT],
    anomaly_result_t *result
);

#endif