#ifndef SENSOR_H
#define SENSOR_H
#include "../Common/types.h"
typedef struct { float temperature,vibration,pressure,current,humidity; uint32_t timestamp_ms; } sensor_sample_t;
status_t sensor_validate(const sensor_sample_t *sample);
void sensor_simulate(sensor_sample_t *sample, uint32_t index, bool fault);
#endif
