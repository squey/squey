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

#ifndef PVCORE_PVALGORITHMS_H
#define PVCORE_PVALGORITHMS_H

#include <cmath>
#include <limits>
#include <algorithm>
#include <iterator>
#include <pvkernel/core/inendi_intrin.h>

#include <QtCore/qglobal.h>

namespace PVCore
{

template <typename T>
T clamp(const T& value, const T& low, const T& high)
{
	return value < low ? low : (value > high ? high : value);
}

/**
 * Compute the upper power of 2 of a value.
 *
 * This is an adaptation for 64 bits number of the algorithm found at:
 * http://www.gamedev.net/topic/229831-nearest-power-of-2/page__p__2494431#entry2494431
 *
 * @param v the value from which we want the upper power of 2
 */

inline uint64_t upper_power_of_2(uint64_t v)
{
	constexpr static uint64_t MantissaMask = (1UL << 52) - 1;

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

	(*(double*)&v) = (double)v;
	v = (v + MantissaMask) & ~MantissaMask;
	return (uint64_t)(*(double*)&v);

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif
}

/**
 * Map a value to a natural logarithm scale
 *
 * @param value the value in [a;b]
 * @param a the lower bound value
 * @param b the upper bound value
 *
 * @return the scale in [0;1]
 */
template <class T>
inline T log_scale(const T value, const T a, const T b)
{
	return std::log((value - a) + 1.) / std::log((b - a) + 1.);
}

/**
 * Map a value from a natural logarithm scale
 *
 * @param value the value in [a;b]
 * @param a the lower bound value
 * @param b the upper bound value
 *
 * @return the scale in [0;1]
 */
template <class T>
inline T inv_log_scale(const T value, const T a, const T b)
{
	return (std::exp(std::log((b - a) + 1.) * value) + a) - 1.;
}

/**
 * Test if a positive integer is a power of two
 *
 * SSE 4.2 powered or fallback to
 *http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
 *
 * @param v the value to test
 *
 * return true is v is a power of 2, false otherwise
 */
inline bool is_power_of_two(uint32_t v)
{
#ifdef __SSE4_2__
	return (_mm_popcnt_u32(v) == 1);
#else
	return (v && !(v & (v - 1)));
#endif
}

/**
 * revert effect from commit 7be751d2d63ae3c52ca16041ee3e46146cd18d62 "Inverse
 * plotting values (because we all fail during demonstrations...)".
 *
 * @param value the value to revert
 *
 * @return the inverted vale
 */
inline uint32_t invert_plotting_value(uint32_t value)
{
	return ~(value);
}

/**
 * revert effect from commit 7be751d2d63ae3c52ca16041ee3e46146cd18d62 "Inverse
 * plotting values (because we all fail during demonstrations...)".
 *
 * qreal version.
 *
 * @param value the value to revert
 *
 * @return the inverted vale
 */
inline uint32_t invert_plotting_value(qreal value)
{
	return ~((uint32_t)clamp(value, 0., (qreal)std::numeric_limits<uint32_t>::max()));
}
} // namespace PVCore

#endif
