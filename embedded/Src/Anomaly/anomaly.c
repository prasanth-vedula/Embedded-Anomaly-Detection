#include "../Inc/Anomaly/anomaly.h"
#include "../Inc/Config/config.h"
#include "../Inc/Config/deployment_parameters.h"
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#define FAULT_CONTRIBUTION_RATIO (0.20F)
#define FAULT_MASK_EPSILON       (1.0e-6F)

static uint32_t persistence;

static uint32_t calculate_fault_mask(
    const float feature_error[ANOMALY_FEATURE_COUNT],
    float total_score
)
{
    if (feature_error == NULL ||
        !isfinite(total_score) ||
        total_score <= 0.0F) {
        return 0U;
    }

    uint32_t mask = 0U;

    /*
     * Each feature contributes a fraction of the total reconstruction
     * error. A sensor is marked as contributing when its share reaches
     * the equal-share baseline:
     *
     *     1 / FEATURE_COUNT = 20%
     *
     * The epsilon prevents harmless floating-point boundary effects
     * from changing the attribution result.
     */
    for (uint32_t feature = 0U;
         feature < ANOMALY_FEATURE_COUNT;
         ++feature) {

        const float contribution_ratio =
            feature_error[feature] / total_score;

        if (contribution_ratio + FAULT_MASK_EPSILON >=
            FAULT_CONTRIBUTION_RATIO) {

            mask |= (1UL << feature);
        }
    }

    return mask;
}

void anomaly_init(void)
{
    persistence = 0U;
}

status_t anomaly_update(
    float score,
    const float feature_error[ANOMALY_FEATURE_COUNT],
    anomaly_result_t *result
)
{
    if (result == NULL ||
        feature_error == NULL) {
        return STATUS_INVALID_ARG;
    }

    if (!isfinite(score) ||
        score < 0.0F) {
        return STATUS_INVALID_ARG;
    }

    result->score = score;

    for (uint32_t i = 0U;
         i < ANOMALY_FEATURE_COUNT;
         ++i) {

        if (!isfinite(feature_error[i]) ||
            feature_error[i] < 0.0F) {
            return STATUS_INVALID_DATA;
        }

        result->feature_error[i] =
            feature_error[i];
    }

    result->fault_mask =
        calculate_fault_mask(
            feature_error,
            score
        );

    if (score >= deployment_anomaly_threshold) {

        if (persistence < UINT32_MAX) {
            ++persistence;
        }

    } else {
        persistence = 0U;
    }

    result->persistence = persistence;

    if (persistence >= CONFIRMATION_COUNT) {

        result->state = ANOMALY_CONFIRMED;

    } else if (persistence > 0U) {

        result->state = ANOMALY_SUSPECT;

    } else {

        result->state = ANOMALY_NORMAL;
    }

    return STATUS_OK;
}