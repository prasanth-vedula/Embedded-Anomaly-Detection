#include <math.h>
#include <stddef.h>

#include "../Inc/Preprocess/preprocess.h"
#include "../Inc/Config/deployment_parameters.h"

status_t preprocess_sample(
    const sensor_sample_t *sample,
    float out[5]
)
{
    if (sample == NULL || out == NULL) {
        return STATUS_INVALID_ARG;
    }

    if (sensor_validate(sample) != STATUS_OK) {
        return STATUS_INVALID_DATA;
    }

    const float values[5] = {
        sample->temperature,
        sample->vibration,
        sample->pressure,
        sample->current,
        sample->humidity
    };

    for (uint32_t i = 0U; i < 5U; ++i) {
        out[i] =
            (
                values[i] -
                deployment_sensor_mean[i]
            ) /
            deployment_sensor_scale[i];

        if (!isfinite(out[i])) {
            return STATUS_ERROR;
        }
    }

    return STATUS_OK;
}