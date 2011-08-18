#ifndef PVKERNEL_CORE_PICVIZ_INTRIN_H
#define PVKERNEL_CORE_PICVIZ_INTRIN_H

#ifdef WIN32
#include <smmintrin.h>
#else
#include <immintrin.h>
#endif

bool has_sse41();
bool has_sse42();
void init_cpuid();


#endif
