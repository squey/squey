/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PARALLELVIEW_PVTLRBUFFER_H
#define PARALLELVIEW_PVTLRBUFFER_H

#include <pvparallelview/common.h>

namespace PVParallelView
{

template <size_t Bbits = PARALLELVIEW_ZZT_BBITS>
class PVTLRBuffer
{
  public:
	constexpr static size_t length = 3 * (1 << (2 * Bbits));

  public:
	struct index_t {
		explicit index_t(uint32_t vv = 0U) { v = vv; }

		index_t(uint32_t t, uint32_t l, uint32_t r) { v = (t << (2 * Bbits)) + (l << Bbits) + r; }

		union {
			uint32_t v;
			struct {
				uint32_t r : Bbits;
				uint32_t l : Bbits;
				uint32_t t : 2;
			} s;
		};
	};

  public:
	PVTLRBuffer() { clear(); }

	void clear() { memset(_data, -1, length * sizeof(uint32_t)); }

	const uint32_t& operator[](size_t i) const { return _data[i]; }

	uint32_t& operator[](size_t i) { return _data[i]; }

	uint32_t* get_data() { return _data; }

  private:
	uint32_t _data[length];
};
} // namespace PVParallelView

#endif // PARALLELVIEW_PVTLRBUFFER_H
