#ifndef tests_common_h
#define tests_common_h

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))
#define B_SET(x, n)      ((x) |= (1<<(n)))

#include "point_buffer.h"
#include "collision_buf.h"

#endif
