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

#ifndef PVCORE_PVBITVISITOR_H
#define PVCORE_PVBITVISITOR_H

#include <pvkernel/core/squey_intrin.h>

#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/type_traits/function_traits.hpp>

#include <limits>

namespace PVCore
{

namespace PVBitVisitor
{

namespace __impl
{

template <typename T, typename F>
class visit_bits_base
{
  protected:
	// Ensure that T is an unsigned integer type
	static_assert(std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,
	              "PVCore::PVBitVisitor::visit_bit, T must be an usigned integral type");

	// AG: FIXME
	// typedef typename boost::function_traits<F>::arg0_type bit_t;
	typedef size_t bit_t;
	static_assert(std::numeric_limits<bit_t>::is_integer,
	              "First argument of F must be an integer type");

	static constexpr bit_t nbits = std::numeric_limits<T>::digits;
};

template <typename T, typename F>
struct visit_bits : private visit_bits_base<T, F> {
  private:
	typedef visit_bits_base<T, F> base_t;
	typedef typename base_t::bit_t bit_t;

  public:
	static void f(const T chunk, F const& f, const bit_t offset)
	{
		if (chunk == 0) {
			// If chunk is empty, just do nothing
			return;
		}

		if (chunk == boost::integer_traits<T>::const_max) {
			// If chunk is full, then go with all of the bits
			for (bit_t b = offset; b < offset + base_t::nbits; b++) {
				f(b);
			}
			return;
		}

		// Go one granularity under
		typedef boost::uint_t<base_t::nbits / 2> uint_lower_traits;
		typedef typename uint_lower_traits::exact uint_lower_t;
		visit_bits<uint_lower_t, F>::f(chunk, f, offset);
		visit_bits<uint_lower_t, F>::f(chunk >> (base_t::nbits / 2), f, offset + base_t::nbits / 2);
	}
};

// Finest granulairty is the byte
template <typename F>
struct visit_bits<uint8_t, F> : private visit_bits_base<uint8_t, F> {
  private:
	typedef visit_bits_base<uint8_t, F> base_t;
	typedef typename base_t::bit_t bit_t;

  public:
	static void f(const uint8_t chunk, F const& f, const bit_t offset)
	{
		if (chunk == 0) {
			return;
		}
		if (chunk == 0xFF) {
			for (bit_t b = offset; b < offset + 8; b++) {
				f(b);
			}
			return;
		}

		for (bit_t b = 0; b < 8; b++) {
			if (chunk & (1 << b)) {
				f(b + offset);
			}
		}
	}
};

#ifdef __SSE4_1__
// Specialisation if SSE4.1 is enabled

#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <typename F>
struct visit_bits<__m128i, F> {
  private:
	typedef size_t bit_t;

  public:
	static void f(const __m128i chunk, F const& f, const bit_t offset)
	{
		const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
		if (_mm_testz_si128(chunk, ones) == 1) {
			// If chunk is empty, just do nothing
			return;
		}

		if (_mm_testc_si128(_mm_setzero_si128(), chunk) == 1) {
			// If chunk is full, then go with all of the bits
			for (bit_t b = offset; b < offset + 128; b++) {
				f(b);
			}
			return;
		}

		// Go one granularity under
		const uint64_t v1 = _mm_extract_epi64(chunk, 0);
		const uint64_t v2 = _mm_extract_epi64(chunk, 1);
		visit_bits<uint64_t, F>::f(v1, f, offset);
		visit_bits<uint64_t, F>::f(v2, f, offset + 64);
	}
};

#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

#endif
} // namespace __impl

template <typename T, typename F>
inline void visit_bits(const T chunk, F const& f, const size_t offset = 0)
{
	// FIXME: size_t should be determined from f
	__impl::visit_bits<T, F>::f(chunk, f, offset);
}
} // namespace PVBitVisitor
} // namespace PVCore

#endif
