#ifndef EVENT_H
#define EVENT_H

#include "../Anomaly/anomaly.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint32_t timestamp_ms;
    float score;
    anomaly_state_t severity;
    uint32_t fault_mask;
} anomaly_event_t;

bool event_create(
    uint32_t timestamp_ms,
    const anomaly_result_t *result,
    anomaly_event_t *event
);

#endif