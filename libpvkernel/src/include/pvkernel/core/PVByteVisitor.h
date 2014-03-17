/**
 * \file PVByteVisitor.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVCORE_PVBYTEVISITOR_H
#define PVCORE_PVBYTEVISITOR_H

#include <pvkernel/core/picviz_intrin.h>

#include <boost/integer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/type_traits/function_traits.hpp>

#include <limits>

namespace PVCore {

namespace PVByteVisitor {

namespace __impl {

template <typename T, typename F>
class visit_bytes_base
{
protected:
	// Ensure that T is an unsigned integer type
	static_assert(std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed, "PVCore::PVBitVisitor::visit_bit, T must be an usigned integral type");

	// AG: FIXME
	//typedef typename boost::function_traits<F>::arg0_type byte_t;
	typedef size_t byte_t;
	static_assert(std::numeric_limits<byte_t>::is_integer, "First argument of F must be an integer type");

	static constexpr byte_t nbytes = sizeof(T);
};

template <typename T, typename F>
struct visit_bytes: private visit_bytes_base<T, F>
{
private:
	typedef visit_bytes_base<T, F> base_t;
	typedef typename base_t::byte_t byte_t;
public:
	static void f(const T chunk, F const& f, const byte_t offset)
	{
		if (chunk == 0) {
			// If chunk is empty, just do nothing
			return;
		}

		if (chunk == boost::integer_traits<T>::const_max) {
			// If chunk is full, then go with all of the bytes
			for (byte_t b = offset; b < offset+base_t::nbytes; b++) {
				f(b);
			}
			return;
		}

		// Go one granularity under
		typedef boost::uint_t<(base_t::nbytes*8)/2> uint_lower_traits;
		typedef typename uint_lower_traits::exact uint_lower_t;
		visit_bytes<uint_lower_t, F>::f(chunk, f, offset);
		visit_bytes<uint_lower_t, F>::f(chunk>>((base_t::nbytes*8)/2), f, offset + base_t::nbytes/2);
	}
};

// Finest granulairty is the byte
template <typename F>
struct visit_bytes<uint32_t, F>: private visit_bytes_base<uint32_t, F>
{
private:
	typedef visit_bytes_base<uint32_t, F> base_t;
	typedef typename base_t::byte_t byte_t;
public:
	static inline void f(const uint32_t chunk, F const& f, const byte_t offset)
	{
		if (chunk == 0) {
			return;
		}

		for (byte_t i = 0;  i < 4; i++) {
			if (chunk & (0xFFU << (i<<3))) {
				f(i+offset);
			}
		}
	}
};

#ifdef __SSE4_1__
// Specialisation if SSE4.1 is enabled
template <typename F>
struct visit_bytes<__m128i, F>
{
private:
	typedef size_t byte_t;
public:
	static void f(const __m128i chunk, F const& f, const byte_t offset)
	{
		const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
		if (_mm_testz_si128(chunk, ones) == 1) {
			// If chunk is empty, just do nothing
			return;
		}

		if (_mm_testc_si128(_mm_setzero_si128(), chunk) == 1) {
			// If chunk is full, then go with all of the bytes
			for (byte_t b = offset; b < offset+16; b++) {
				f(b);
			}
			return;
		}

		// Go one granularity under
		const uint64_t v1 = _mm_extract_epi64(chunk, 0);
		const uint64_t v2 = _mm_extract_epi64(chunk, 1);
		visit_bytes<uint64_t, F>::f(v1, f, offset);
		visit_bytes<uint64_t, F>::f(v2, f, offset+8);
	}
};
#endif


uint8_t const* get_nth_slice_serial(uint8_t const* buffer, size_t sbuf, size_t n, size_t& size_ret);
uint8_t const* get_nth_slice_sse(uint8_t const* buffer, size_t sbuf, size_t n, size_t& size_ret, bool* complete = nullptr);

} // __impl

template <typename T, typename F>
inline void visit_bytes(const T chunk, F const& f, const size_t offset = 0)
{
	// FIXME: size_t should be determined from f
	__impl::visit_bytes<T, F>::f(chunk, f, offset);
}

inline uint8_t const* get_nth_slice(uint8_t const* buffer, size_t sbuf, size_t n, size_t& size_ret, bool* complete = nullptr) { return __impl::get_nth_slice_sse(buffer, sbuf, n, size_ret, complete); }

// Get back the n-th slice, that is, if a buffer is like this:
// [data 0] \0 [data 1] \0,
// the 0'th slice is data 0, the "1-th" slice is data 1, etc...
template <typename F>
bool visit_nth_slice(uint8_t const* buffer, size_t sbuf, size_t n, F const& f)
{
	size_t size_final;
	uint8_t const* const buf_start = get_nth_slice(buffer, sbuf, n, size_final);
	if (buf_start) {
		f(buf_start, size_final);
		return true;
	}
	return false;
}

}

}

#endif
