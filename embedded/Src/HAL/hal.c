#include <stdio.h>
#include "../Inc/HAL/hal.h"
static uint32_t tick;
void hal_init(void){tick=0U;}
void hal_log(const char *level,const char *message){(void)printf("[%s] %s\n",level,message);}
uint32_t hal_tick_ms(void){return tick++;}
