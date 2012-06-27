#ifndef PVCORE_PVBITCOUNT_H
#define PVCORE_PVBITCOUNT_H

#include <pvkernel/core/general.h>

namespace PVCore {

#ifdef __SSE4_1__
inline size_t bit_count_u64(const uint64_t v) { return _mm_popcnt_u64(v); }
inline size_t bit_count_u32(const uint32_t v) { return _mm_popcnt_u32(v); }
#else
size_t bit_count_u32(uint32_t v)
{
	v = v - ((v >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}
size_t bit_count_u64(uint64_t v)
{
	v = v - ((v >> 1) & 0x5555555555555555ULL);
	v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
	return (((v + (v >> 4)) & 0x0f0f0f0f0f0f0f0f) * 0x0101010101010101) >> 56;
}
#endif

#endif
