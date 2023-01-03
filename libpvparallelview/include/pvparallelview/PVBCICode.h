/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PARALLELVIEW_PVBCICODE_H
#define PARALLELVIEW_PVBCICODE_H

#include <stdint.h>
#include <pvparallelview/common.h>
#include <tbb/cache_aligned_allocator.h>

namespace PVParallelView
{

/**
 * It represents a line in a Zone of ParallelView
 *
 * * Line is:
 *     * position (listing position)
 *     * left plotting value
 *     * right plotting value
 *     * color
 *     * direction (use?)
 */
template <size_t Bbits = NBITS_INDEX>
struct PVBCICode {
	static_assert((Bbits >= 1) & (Bbits <= 11), "PVBCICode: Bbits must be between 1 and 11.");

	typedef tbb::cache_aligned_allocator<PVBCICode> allocator;
	union {
		uint64_t int_v;
		struct {
			uint32_t idx;
			uint32_t l : Bbits;
			uint32_t r : Bbits;
			uint32_t color : 8;
			uint32_t type : 2;
		} __attribute((packed)) s;
	};

	typedef enum { STRAIGHT = 0, UP = 1, DOWN = 2 } _type_t;

	static PVBCICode* allocate_codes(size_t n)
	{
		PVBCICode* ret = PVBCICode::allocator().allocate(n);
		return ret;
	}

	static void free_codes(PVBCICode* codes) { PVBCICode::allocator().deallocate(codes, 0); }
};

typedef union {
	PVBCICode<10> as_10;
	PVBCICode<11> as_11;
} PVBCICodeBase;
} // namespace PVParallelView

#endif
