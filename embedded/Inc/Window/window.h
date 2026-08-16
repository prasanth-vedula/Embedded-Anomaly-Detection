#ifndef WINDOW_H
#define WINDOW_H
#include "../Common/types.h"
#include "../Config/config.h"
typedef struct { float values[WINDOW_LENGTH][FEATURE_COUNT]; uint32_t count; } window_t;
void window_init(window_t *w); status_t window_push(window_t *w,const float features[FEATURE_COUNT]); bool window_ready(const window_t *w); const float *window_data(const window_t *w); void window_reset(window_t *w);
#endif
