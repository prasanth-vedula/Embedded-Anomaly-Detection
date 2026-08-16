#include <string.h>
#include "../Inc/Window/window.h"
void window_init(window_t *w){if(w){(void)memset(w,0,sizeof(*w));}}
status_t window_push(window_t *w,const float f[FEATURE_COUNT]){if(!w||!f)return STATUS_INVALID_ARG;if(w->count<WINDOW_LENGTH){(void)memcpy(w->values[w->count],f,sizeof(w->values[0]));w->count++;return STATUS_OK;} (void)memmove(w->values[0],w->values[1],(WINDOW_LENGTH-1U)*sizeof(w->values[0]));(void)memcpy(w->values[WINDOW_LENGTH-1U],f,sizeof(w->values[0]));return STATUS_OK;}
bool window_ready(const window_t *w){return w!=0&&w->count>=WINDOW_LENGTH;} const float *window_data(const window_t *w){return w?&w->values[0][0]:0;} void window_reset(window_t *w){window_init(w);}
