#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "../Common/types.h"
#include "../Sensor/sensor.h"

status_t preprocess_sample(
    const sensor_sample_t *sample,
    float out[5]
);

#endif