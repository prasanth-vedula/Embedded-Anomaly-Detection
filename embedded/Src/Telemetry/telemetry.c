#include <stdio.h>
#include "../Inc/Telemetry/telemetry.h"
void telemetry_init(void){}
void telemetry_publish(const anomaly_event_t *e){if(e)(void)printf("{\"ts\":%u,\"score\":%.6f,\"severity\":%d,\"fault_mask\":%u}\n",e->timestamp_ms,(double)e->score,(int)e->severity,e->fault_mask);}
