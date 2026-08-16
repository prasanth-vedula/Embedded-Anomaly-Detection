#include <stdint.h>

#include "../Inc/HAL/hal.h"
#include "../Inc/Sensor/sensor.h"
#include "../Inc/Preprocess/preprocess.h"
#include "../Inc/Window/window.h"
#include "../Inc/TinyML/tinyml.h"
#include "../Inc/Anomaly/anomaly.h"
#include "../Inc/Event/event.h"
#include "../Inc/Telemetry/telemetry.h"

int main(void)
{
    hal_init();

    sensor_sample_t sensor;
    float features[FEATURE_COUNT];

    window_t window;
    anomaly_result_t anomaly;
    anomaly_event_t event;

    window_init(&window);

    if (tinyml_init() != STATUS_OK) {
        return 1;
    }

    anomaly_init();
    telemetry_init();

    for (uint32_t i = 0U; i < 240U; ++i) {

        sensor_simulate(
            &sensor,
            i,
            i > 180U
        );

        if (preprocess_sample(
                &sensor,
                features
            ) != STATUS_OK) {
            continue;
        }

        if (window_push(
                &window,
                features
            ) != STATUS_OK) {
            continue;
        }

        if (!window_ready(&window)) {
            continue;
        }

        float score = 0.0F;
        float feature_error[
            TINYML_FEATURE_COUNT
        ];

        if (tinyml_infer_with_contributions(
                window_data(&window),
                &score,
                feature_error
            ) != STATUS_OK) {
            continue;
        }

        if (anomaly_update(
                score,
                feature_error,
                &anomaly
            ) != STATUS_OK) {
            continue;
        }

        if (event_create(
                sensor.timestamp_ms,
                &anomaly,
                &event
            )) {
            telemetry_publish(&event);
        }
    }

    return 0;
}