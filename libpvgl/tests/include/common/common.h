/**
 * \file common.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef tests_common_h
#define tests_common_h

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))
#define B_SET(x, n)      ((x) |= (1<<(n)))

#define VERIFY(v) __VERIFY(v, __FILE__, __LINE__)
#define __VERIFY(v,F,L)\
	if (!(v)) {\
		std::cerr << F << ":" << L << " :" << #v << " isn't valid." << std::endl;\
		abort();\
	}

#include "point_buffer.h"
#include "collision_buf.h"

#endif
