#include <assert.h>
#include <math.h>
#include <stdint.h>

#include "../../embedded/Inc/TinyML/tinyml.h"

int main(void)
{
    assert(
        tinyml_init()
        == STATUS_OK
    );

    float input[TINYML_INPUT_SIZE];

    for (uint32_t i = 0U;
         i < TINYML_INPUT_SIZE;
         ++i) {

        input[i] =
            ((float)((int32_t)(i % 11U) - 5))
            * 0.1F;
    }

    float score_a = 0.0F;

    float contributions_a[
        TINYML_FEATURE_COUNT
    ];

    assert(
        tinyml_infer_with_contributions(
            input,
            &score_a,
            contributions_a
        )
        == STATUS_OK
    );

    assert(isfinite(score_a));
    assert(score_a >= 0.0F);

    float contribution_sum = 0.0F;

    for (uint32_t i = 0U;
         i < TINYML_FEATURE_COUNT;
         ++i) {

        assert(
            isfinite(
                contributions_a[i]
            )
        );

        assert(
            contributions_a[i] >= 0.0F
        );

        contribution_sum +=
            contributions_a[i];
    }

    assert(
        fabsf(
            score_a -
            (
                contribution_sum /
                (float)TINYML_FEATURE_COUNT
            )
        )
        < 1.0e-4F
    );

    /*
     * Inference must be deterministic.
     */
    float score_b = 0.0F;

    float contributions_b[
        TINYML_FEATURE_COUNT
    ];

    assert(
        tinyml_infer_with_contributions(
            input,
            &score_b,
            contributions_b
        )
        == STATUS_OK
    );

    assert(
        fabsf(score_a - score_b)
        < 1.0e-6F
    );

    for (uint32_t i = 0U;
         i < TINYML_FEATURE_COUNT;
         ++i) {

        assert(
            fabsf(
                contributions_a[i] -
                contributions_b[i]
            )
            < 1.0e-6F
        );
    }

    /*
     * The legacy API must produce the same anomaly score.
     */
    float legacy_score = 0.0F;

    assert(
        tinyml_infer(
            input,
            &legacy_score
        )
        == STATUS_OK
    );

    assert(
        fabsf(
            score_a -
            legacy_score
        )
        < 1.0e-6F
    );

    /*
     * Invalid arguments.
     */
    assert(
        tinyml_infer(
            NULL,
            &legacy_score
        )
        == STATUS_INVALID_ARG
    );

    assert(
        tinyml_infer(
            input,
            NULL
        )
        == STATUS_INVALID_ARG
    );

    float invalid_input[
        TINYML_INPUT_SIZE
    ];

    for (uint32_t i = 0U;
         i < TINYML_INPUT_SIZE;
         ++i) {
        invalid_input[i] = 0.0F;
    }

    invalid_input[17] = NAN;

    assert(
        tinyml_infer(
            invalid_input,
            &legacy_score
        )
        == STATUS_INVALID_DATA
    );

    return 0;
}