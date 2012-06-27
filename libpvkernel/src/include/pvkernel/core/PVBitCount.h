#ifndef PVCORE_PVBITCOUNT_H
#define PVCORE_PVBITCOUNT_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/picviz_intrin.h>

namespace PVCore {

namespace PVBitCount {

#ifdef __SSE4_1__
inline size_t bit_count(const uint64_t v) { return _mm_popcnt_u64(v); }
inline size_t bit_count(const uint32_t v) { return _mm_popcnt_u32(v); }
#else
inline size_t bit_count(uint32_t v)
{
	v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}
inline size_t bit_count(uint64_t v)
{
	v = v - ((v >> 1) & 0x5555555555555555ULL);
	v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
	return (((v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;
}
#endif

}

}

#endif
