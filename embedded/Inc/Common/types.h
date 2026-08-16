#ifndef TYPES_H
#define TYPES_H
#include <stdbool.h>
#include <stdint.h>
typedef enum { STATUS_OK=0, STATUS_INVALID_ARG=1, STATUS_INVALID_DATA=2, STATUS_NOT_READY=3, STATUS_ERROR=4 } status_t;
#endif
