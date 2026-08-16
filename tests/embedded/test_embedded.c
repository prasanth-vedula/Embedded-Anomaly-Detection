#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "../../embedded/Inc/Common/types.h"
#include "../../embedded/Inc/Config/config.h"
#include "../../embedded/Inc/Sensor/sensor.h"
#include "../../embedded/Inc/Preprocess/preprocess.h"
#include "../../embedded/Inc/Window/window.h"
#include "../../embedded/Inc/TinyML/tinyml.h"
#include "../../embedded/Inc/Anomaly/anomaly.h"
#include "../../embedded/Inc/Event/event.h"

int main(void)
{
    sensor_sample_t sensor = {
        68.0F,
        0.42F,
        4.8F,
        8.2F,
        45.0F,
        0U
    };

    float features[FEATURE_COUNT];

    assert(
        sensor_validate(&sensor)
        == STATUS_OK
    );

    sensor.humidity = 101.0F;

    assert(
        sensor_validate(&sensor)
        == STATUS_INVALID_DATA
    );

    sensor.humidity = 45.0F;

    assert(
        preprocess_sample(
            &sensor,
            features
        )
        == STATUS_OK
    );

    window_t window;

    window_init(&window);

    for (uint32_t i = 0U;
         i < WINDOW_LENGTH;
         ++i) {

        assert(
            window_push(
                &window,
                features
            )
            == STATUS_OK
        );
    }

    assert(window_ready(&window));

    assert(
        tinyml_init()
        == STATUS_OK
    );

    float score = 0.0F;

    float feature_error[
        TINYML_FEATURE_COUNT
    ];

    assert(
        tinyml_infer_with_contributions(
            window_data(&window),
            &score,
            feature_error
        )
        == STATUS_OK
    );

    assert(isfinite(score));
    assert(score >= 0.0F);

    float contribution_sum = 0.0F;

    for (uint32_t i = 0U;
         i < TINYML_FEATURE_COUNT;
         ++i) {

        assert(isfinite(feature_error[i]));
        assert(feature_error[i] >= 0.0F);

        contribution_sum +=
            feature_error[i];
    }

    /*
     * TinyML defines the global reconstruction error as
     * the mean of the five sensor reconstruction errors.
     */
    assert(
        fabsf(
            score -
            (
                contribution_sum /
                (float)TINYML_FEATURE_COUNT
            )
        ) < 1.0e-4F
    );

    /*
     * Verify anomaly persistence independently from
     * the particular model attribution produced above.
     *
     * We deliberately use a contribution vector with one
     * dominant sensor. The current anomaly engine should
     * identify that sensor without a simulator fault flag.
     */
    anomaly_result_t result;

    anomaly_init();

    const float fault_contribution[
        ANOMALY_FEATURE_COUNT
    ] = {
        0.01F,
        0.01F,
        0.01F,
        1.46F,
        0.01F
    };

    const float fault_score =
        (
            fault_contribution[0] +
            fault_contribution[1] +
            fault_contribution[2] +
            fault_contribution[3] +
            fault_contribution[4]
        )
        /
        (float)ANOMALY_FEATURE_COUNT;

    assert(
        fault_score > 0.0F
    );

    assert(
        anomaly_update(
            fault_score,
            fault_contribution,
            &result
        )
        == STATUS_OK
    );

    assert(
        result.state == ANOMALY_SUSPECT
    );

    /*
     * Current-fault contribution is feature index 3:
     *
     * bit 3 = 0x08 = 8
     */
    assert(
        result.fault_mask == 8U
    );

    assert(
        anomaly_update(
            fault_score,
            fault_contribution,
            &result
        )
        == STATUS_OK
    );

    assert(
        anomaly_update(
            fault_score,
            fault_contribution,
            &result
        )
        == STATUS_OK
    );

    assert(
        result.state == ANOMALY_CONFIRMED
    );

    assert(
        result.persistence == 3U
    );

    assert(
        result.fault_mask == 8U
    );

    anomaly_event_t event;

    assert(
        event_create(
            1234U,
            &result,
            &event
        )
    );

    assert(
        event.timestamp_ms == 1234U
    );

    assert(
        event.severity == ANOMALY_CONFIRMED
    );

    assert(
        event.fault_mask == 8U
    );

    assert(event.score > 0.0F);

    /*
     * A confirmed anomaly without any contributing
     * sensor must not produce an event.
     */
    result.fault_mask = 0U;

    assert(
        !event_create(
            1234U,
            &result,
            &event
        )
    );

    return 0;
}