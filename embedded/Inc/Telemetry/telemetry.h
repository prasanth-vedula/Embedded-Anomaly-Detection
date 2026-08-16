#ifndef TELEMETRY_H
#define TELEMETRY_H
#include "../Event/event.h"
void telemetry_init(void); void telemetry_publish(const anomaly_event_t *event);
#endif
