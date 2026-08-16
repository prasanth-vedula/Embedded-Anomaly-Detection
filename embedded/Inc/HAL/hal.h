#ifndef HAL_H
#define HAL_H
#include <stdint.h>
void hal_init(void); void hal_log(const char *level,const char *message); uint32_t hal_tick_ms(void);
#endif
