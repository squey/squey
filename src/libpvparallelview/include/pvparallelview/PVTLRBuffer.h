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
