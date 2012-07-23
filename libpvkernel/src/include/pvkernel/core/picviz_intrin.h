/**
 * \file picviz_intrin.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVKERNEL_CORE_PICVIZ_INTRIN_H
#define PVKERNEL_CORE_PICVIZ_INTRIN_H

#ifdef WIN32
#define __SSE4_1__
#include <smmintrin.h>
#else
#include <immintrin.h>
#endif

#include <pvkernel/core/general.h>

namespace PVCore {

class LibKernelDecl PVIntrinsics
{
public:
	static bool has_sse41();
	static bool has_sse42();
	static void init_cpuid();

private:
	static bool _has_sse41;
	static bool _has_sse42;
	static bool _init_done;
};

}


#endif
