#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "../../embedded/Inc/Common/types.h"
#include "../../embedded/Inc/Config/config.h"
#include "../../embedded/Inc/Config/deployment_parameters.h"
#include "../../embedded/Inc/Sensor/sensor.h"
#include "../../embedded/Inc/Preprocess/preprocess.h"

int main(void)
{
    /*
     * The embedded preprocessing reference is not the simulator's
     * nominal value. It is the exact mean exported from Python training.
     *
     * Therefore this test constructs a sample directly from the
     * exported deployment means and verifies that the embedded
     * preprocessing produces approximately zero.
     */
    sensor_sample_t sample = {
        deployment_sensor_mean[0],
        deployment_sensor_mean[1],
        deployment_sensor_mean[2],
        deployment_sensor_mean[3],
        deployment_sensor_mean[4],
        0U
    };

    float output[FEATURE_COUNT];

    assert(
        preprocess_sample(
            &sample,
            output
        ) == STATUS_OK
    );

    for (uint32_t i = 0U;
         i < FEATURE_COUNT;
         ++i) {

        assert(isfinite(output[i]));

        assert(
            fabsf(output[i]) < 1.0e-5F
        );
    }

    /*
     * Verify that the exported scale parameters are actually used.
     *
     * mean + scale must normalize to approximately +1.
     */
    sample.temperature =
        deployment_sensor_mean[0] +
        deployment_sensor_scale[0];

    sample.vibration =
        deployment_sensor_mean[1] +
        deployment_sensor_scale[1];

    sample.pressure =
        deployment_sensor_mean[2] +
        deployment_sensor_scale[2];

    sample.current =
        deployment_sensor_mean[3] +
        deployment_sensor_scale[3];

    sample.humidity =
        deployment_sensor_mean[4] +
        deployment_sensor_scale[4];

    assert(
        preprocess_sample(
            &sample,
            output
        ) == STATUS_OK
    );

    for (uint32_t i = 0U;
         i < FEATURE_COUNT;
         ++i) {

        assert(isfinite(output[i]));

        assert(
            fabsf(output[i] - 1.0F) < 1.0e-5F
        );
    }

    /*
     * Verify that invalid sensor data is rejected.
     */
    sample.temperature = NAN;

    assert(
        preprocess_sample(
            &sample,
            output
        ) == STATUS_INVALID_DATA
    );

    /*
     * Verify NULL argument handling.
     */
    assert(
        preprocess_sample(
            NULL,
            output
        ) == STATUS_INVALID_ARG
    );

    assert(
        preprocess_sample(
            &sample,
            NULL
        ) == STATUS_INVALID_ARG
    );

    return 0;
}