/**
 * \file picviz_libdivide.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_LIBDIVIDE_H
#define PICVIZ_LIBDIVIDE_H

#ifdef __SSE4_1__
#define LIBDIVIDE_USE_SSE2 1
#define LIBDIVIDE_USE_SSE4_1 1
#endif

#include <pvkernel/core/libdivide.h>

#endif
