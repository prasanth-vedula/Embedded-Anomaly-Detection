#include "../Inc/Event/event.h"

bool event_create(
    uint32_t timestamp_ms,
    const anomaly_result_t *result,
    anomaly_event_t *event
)
{
    if (result == NULL || event == NULL) {
        return false;
    }

    if (result->state != ANOMALY_CONFIRMED) {
        return false;
    }

    event->timestamp_ms = timestamp_ms;
    event->score = result->score;
    event->severity = result->state;

    /*
     * Fault mask comes from the actual per-sensor reconstruction
     * contribution calculated by the TinyML inference path.
     */
    event->fault_mask = result->fault_mask;

    return event->fault_mask != 0U;
}