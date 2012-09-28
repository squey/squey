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

}

template <typename T, typename F>
inline void visit_bytes(const T chunk, F const& f, const size_t offset = 0)
{
	// FIXME: size_t should be determined from f
	__impl::visit_bytes<T, F>::f(chunk, f, offset);
}

// Get back the n-th slice, that is, if a buffer is like this:
// [data 0] \0 [data 1] \0,
// the 0'th slice is data 0, the "1-th" slice is data 1, etc...
template <typename F>
void visit_nth_slice(uint8_t const* buffer, size_t sbuf, size_t n, F const& f)
{
	n++;
	// n now is the number of '\0' that we have to look for
	size_t nfound = 0;
	// Algorithm to vectorize:
#if 0
	size_t off_start = 0;
	for (size_t i = 0; i < size; i++) {
		if (buffer[i]) {
			ndone++;
			if (ndone == n) {
				f(&buffer[off_start], i-off_start);
				return;
			}
			off_start=i+1;
		}
	}
#endif
	const size_t sse_sbuf = (sbuf>>4)<<4;
	const __m128i sse_zero = _mm_setzero_si128();
	const __m128i sse_ff = _mm_set1_epi32(0xFFFFFFFF);
	__m128i sse_buf, sse_cmp;
	// TODO: prelogue for alignement!
	size_t off_start = 0;
	size_t i;
	for (i = 0; i < sse_sbuf; i += 16) {
		sse_buf = _mm_loadu_si128((const __m128i*) &buffer[i]);
		sse_cmp = _mm_cmpeq_epi8(sse_buf, sse_zero);
		if (_mm_testz_si128(sse_cmp, sse_ff) == 1) {
			continue;
		}
		// Count the number of bits set, and divide them by 8 !
		const uint64_t b0 = _mm_extract_epi64(sse_cmp, 0);
		const uint64_t b1 = _mm_extract_epi64(sse_cmp, 1); 
		const uint64_t p0 = _mm_popcnt_u64(b0)>>3;
		const uint64_t p1 = _mm_popcnt_u64(b1)>>3;
		const uint64_t total = p0+p1;
		if (nfound + total >= n) {
			// The end is somewhere in here
			if (p0 > 0) {
				for (size_t j = 0; j < sizeof(uint64_t); j++) {
					if ((b0 & (0xFFULL << (j << 3))) != 0) {
						nfound++;
						if (nfound == n) {
							f(&buffer[off_start], (i+j)-off_start);
							return;
						}
						off_start = i+j+1;
					}
				}
			}
			for (size_t j = 0; j < sizeof(uint64_t); j++) {
				if ((b1 & (0xFFULL << (j << 3))) != 0) {
					nfound++;
					if (nfound == n) {
						f(&buffer[off_start], (i+j+sizeof(uint64_t))-off_start);
						return;
					}
					off_start = i+j+sizeof(uint64_t)+1;
				}
			}
		}
		else {
			// Find out the last one to update off_start
			unsigned int last_off;
			uint64_t blast;
			if (p1 > 0) {
				blast = b1;
				last_off = sizeof(uint64_t);
			}
			else {
				blast = b0;
				last_off = 0;
			}
			for (int j = sizeof(uint64_t)-1; j >= 0; j--) {
				// use _mm_extract_epi8 from the sse_cmp register ?
				if ((blast & (0xFFULL << (j << 3))) != 0) {
					off_start = i+j+last_off+1;
					break;
				}
			}
			nfound += total;
		}
	} 
	for (; i < sbuf; i++) {
		if (buffer[i] == 0) {
			nfound++;
			if (nfound == n) {
				f(&buffer[off_start], i-off_start);
				return;
			}
			off_start = i+1;
		}
	}
}

}

}

#endif
