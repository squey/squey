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

#ifndef PVPARALLELVIEW_TESTS_HELPERS_H
#define PVPARALLELVIEW_TESTS_HELPERS_H

#include <pvparallelview/PVBCICode.h>
#include <pvkernel/core/PVHSVColor.h>

#define NUMBER_BCI_PATTERNS 4

namespace PVParallelView
{

template <size_t Bbits>
struct PVBCIPatterns {
	typedef void (*init_func_t)(PVBCICode<Bbits>*, const size_t);
	typedef enum { RANDOM = 0, GRADIENT, UPDOWN, UPDOWN2 } codes_pattern_t;

	static void init_random_codes(PVBCICode<Bbits>* codes, const size_t n)
	{
		for (size_t i = 0; i < n; i++) {
			PVBCICode<Bbits> c;
			c.int_v = 0;
			c.s.idx = rand();
			c.s.l = rand() & constants<Bbits>::mask_int_ycoord;
			c.s.r = rand() & constants<Bbits>::mask_int_ycoord;
			c.s.color = rand() & ((1 << 9) - 1);
			codes[i] = c;
		}
	}

	static void init_gradient_codes(PVBCICode<Bbits>* codes, const size_t n)
	{
		for (size_t i = 0; i < n; i++) {
			PVBCICode<Bbits> c;
			c.int_v = 0;
			c.s.idx = n - i;
			c.s.l = i & (MASK_INT_YCOORD);
			c.s.r = i & (constants<Bbits>::mask_int_ycoord);
			c.s.color = i % ((1 << HSV_COLOR_NBITS_ZONE) * 6);
			codes[i] = c;
		}
	}

	static void init_updown_codes(PVBCICode<Bbits>* codes, const size_t n)
	{
		for (size_t i = 0; i < n; i++) {
			PVBCICode<Bbits> c;
			c.int_v = 0;
			c.s.idx = n - i;
			if (i < 1024) {
				c.s.l = constants<Bbits>::mask_int_ycoord / 2;
				c.s.type = PVBCICode<Bbits>::UP;
				c.s.color = HSV_COLOR_WHITE.h();
			} else if (i < 3072) {
				c.s.l = constants<Bbits>::mask_int_ycoord / 2;
				c.s.type = PVBCICode<Bbits>::DOWN;
				c.s.color = HSV_COLOR_BLACK.h();
			} else {
				c.s.l = constants<Bbits>::mask_int_ycoord / 5;
				c.s.color = i % ((1 << HSV_COLOR_NBITS_ZONE) * 6);
			}
			c.s.r = i & (constants<Bbits>::mask_int_ycoord);
			codes[i] = c;
		}
	}

	static void init_updown2_codes(PVBCICode<Bbits>* codes, const size_t n)
	{
		for (size_t i = 0; i < n; i++) {
			PVBCICode<Bbits> c;
			c.int_v = 0;
			c.s.idx = n - i;
			if (i < 1024) {
				c.s.l = constants<Bbits>::mask_int_ycoord / 2;
				c.s.type = PVBCICode<Bbits>::UP;
			} else if (i < 3072) {
				c.s.l = constants<Bbits>::mask_int_ycoord / 2;
				c.s.type = PVBCICode<Bbits>::DOWN;
			} else {
				c.s.l = constants<Bbits>::mask_int_ycoord / 5;
			}
			c.s.r = i & (constants<Bbits>::mask_int_ycoord);
			c.s.color = i % ((1 << HSV_COLOR_NBITS_ZONE) * 6);
			codes[i] = c;
		}
	}

	static void init_codes_pattern(PVBCICode<Bbits>* codes, const size_t n, int pattern)
	{
		if (pattern < 0 || pattern >= _nfuncs) {
			pattern = 0;
		}
		_funcs[pattern](codes, n);
	}

	static int get_number_patterns() { return NUMBER_BCI_PATTERNS; }

	static int pattern_string_to_id(const char* str)
	{
		for (int i = 0; i < get_number_patterns(); i++) {
			if (strcasecmp(str, _patterns_str[i]) == 0) {
				return i;
			}
		}
		return -1;
	}

	static const char* const* get_patterns_string() { return _patterns_str; }

  private:
	static const init_func_t _funcs[NUMBER_BCI_PATTERNS];
	static const char* _patterns_str[NUMBER_BCI_PATTERNS];
	static const int _nfuncs = NUMBER_BCI_PATTERNS;
};

#ifdef __clang__ // fix warning with clang but would break build with gcc
template <> const PVParallelView::PVBCIPatterns<10>::init_func_t PVParallelView::PVBCIPatterns<10>::_funcs[NUMBER_BCI_PATTERNS];
template <> const PVParallelView::PVBCIPatterns<11>::init_func_t PVParallelView::PVBCIPatterns<11>::_funcs[NUMBER_BCI_PATTERNS];

template <> const char* PVParallelView::PVBCIPatterns<10>::_patterns_str[NUMBER_BCI_PATTERNS];
template <> const char* PVParallelView::PVBCIPatterns<11>::_patterns_str[NUMBER_BCI_PATTERNS];

template <> const int PVParallelView::PVBCIPatterns<10>::_nfuncs;
template <> const int PVParallelView::PVBCIPatterns<11>::_nfuncs;
#endif

#define INIT_STATIC_PATTERNS(BBITS)                                                                          \
	template <>                                                                                              \
	const PVParallelView::PVBCIPatterns<BBITS>::init_func_t                                                  \
	    PVParallelView::PVBCIPatterns<BBITS>::_funcs[NUMBER_BCI_PATTERNS] = {                                \
	        init_random_codes, init_gradient_codes, init_updown_codes, init_updown2_codes};                  \
	template <>                                                                                              \
	const char* PVParallelView::PVBCIPatterns<BBITS>::_patterns_str[NUMBER_BCI_PATTERNS] = {                 \
	    "random", "gardient", "updown", "updown2"};
}

#endif

