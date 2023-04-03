//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvhwloc.h>
#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField, etc
#include <pvkernel/core/PVSerializeObject.h>

#include "pvbase/types.h" // for PVRow, PVROW_INVALID_VALUE, etc

#include <pvcop/core/algo/selection.h> // for bit_count, invert_selection
#include <pvcop/core/array.h>          // for array<>::data_type, array, etc
#include <pvcop/core/impl/bit.h>       // for to_mem_size
#include <pvcop/core/memarray.h>       // for memarray

#include <cstdbool>    // for bool
#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t, uint8_t
#include <cstdlib>     // for rand
#include <cstring>     // for memset
#include <sys/types.h> // for ssize_t
#include <algorithm>   // for find_if
#include <cassert>     // for assert
#include <xmmintrin.h> // for __m128i, _mm_load_si128, etc

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

/******************************************************************************
 *
 * PVCore::PVSelBitField::PVSelBitField()
 *
 *****************************************************************************/
PVCore::PVSelBitField::PVSelBitField(PVRow count) : _selection(count)
{
}

PVCore::PVSelBitField::PVSelBitField(PVSelBitField const& o) : _selection(o._selection.size())
{
	_selection.copy_from(o._selection, 0, o._selection.size());
}

PVCore::PVSelBitField::operator pvcop_selection_t&()
{
	return _selection;
}
PVCore::PVSelBitField::operator const pvcop_selection_t&() const
{
	return _selection;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::bit_count
 *
 *****************************************************************************/
size_t PVCore::PVSelBitField::bit_count(size_t start /*= 0*/, size_t end /*= 0*/) const
{
	if (end == 0) {
		end = _selection.size() - 1;
	}

	return pvcop::core::algo::bit_count(_selection, start, end);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::is_empty
 *
 *****************************************************************************/
bool PVCore::PVSelBitField::is_empty() const
{
	const size_t chunks = chunk_count();
	auto end = _selection.data() + chunks;
	return std::find_if(_selection.data(), end, [](uint64_t chunk) { return chunk != 0; }) == end;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator=(const PVSelBitField& rhs)
{
	_selection.copy_from(rhs._selection, 0, rhs._selection.size());

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator&(const PVSelBitField& rhs) const
{
	assert(chunk_count() == rhs.chunk_count());

	PVCore::PVSelBitField result(*this);
	result &= rhs;
	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator&=(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] &= rhs._selection.data()[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator|
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator|(const PVSelBitField& rhs) const
{
	assert(chunk_count() == rhs.chunk_count());

	PVSelBitField result(*this);
	result |= rhs;

	return result;
}

/******************************************************************************
*
 * PVCore::PVSelBitField::operator|=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator|=(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] |= rhs._selection.data()[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator-(const PVSelBitField& rhs) const
{
	assert(chunk_count() == rhs.chunk_count());

	PVSelBitField result = *this;
	result -= rhs;

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator-=(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] &= ~rhs._selection.data()[i];
	}

	return *this;
}

void PVCore::PVSelBitField::inplace_sub(PVSelBitField const& a, PVSelBitField const& b)
{
	inplace_map(a, b, [](const chunk_t va, const chunk_t vb) { return va & (~vb); });
}

void PVCore::PVSelBitField::inplace_and(PVSelBitField const& a, PVSelBitField const& b)
{
	inplace_map(a, b, [](const chunk_t va, const chunk_t vb) { return va & vb; });
}

size_t PVCore::PVSelBitField::chunk_count() const
{
	return _selection.mem_size() / (CHUNK_SIZE / CHUNK_SIZE_BYTE);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator^(const PVSelBitField& rhs) const
{
	PVSelBitField result = *this;
	result ^= rhs;

	return result;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator^=(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] ^= rhs._selection.data()[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::or_not
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::or_not(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();

	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] |= ~(rhs._selection.data()[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::and_not
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::and_not(const PVSelBitField& rhs)
{
	assert(chunk_count() == rhs.chunk_count());

	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] &= ~(rhs._selection.data()[i]);
	}

	return *this;
}

PVCore::PVSelBitField PVCore::PVSelBitField::operator~() const
{
	PVCore::PVSelBitField res = *this;
	res._selection = ~res._selection;
	return res;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_all
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_all()
{
	select_byte_pattern(0xFF);
}

void PVCore::PVSelBitField::select_random()
{
	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] = (uint64_t)(rand()) | ((uint64_t)(rand()) << 32);
	}
}

void PVCore::PVSelBitField::select_random(const PVRow n)
{
	select_none();

	for (PVRow i = 0; i < n; i++) {
		set_bit_fast(rand() % count());
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_even
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_even()
{
	select_byte_pattern(0xAA);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_none
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_none()
{
	select_byte_pattern(0x00);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_odd
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_odd()
{
	select_byte_pattern(0x55);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_byte_pattern
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_byte_pattern(const unsigned char byte_pattern)
{
	memset(_selection.data(), byte_pattern, _selection.mem_size());
}
/******************************************************************************
 *
 * PVCore::PVSelBitField::select_inverse
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_inverse()
{
	const size_t chunks = chunk_count();
	for (PVRow i = 0; i < chunks; i++) {
		_selection.data()[i] = ~_selection.data()[i];
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::set_line
 *
 *****************************************************************************/
void PVCore::PVSelBitField::set_line(PVRow line_index, bool bool_value)
{
	_selection[line_index] = bool_value;
}

ssize_t PVCore::PVSelBitField::get_last_nonzero_chunk_index(ssize_t starting_chunk,
                                                            ssize_t ending_chunk) const
{
	if (ending_chunk == -1) {
		ending_chunk = chunk_count() - 1;
	}

	if (!_selection.data() || (starting_chunk < 0 || ending_chunk < 0)) {
		return -1;
	}
#ifdef __SSE4_1__
	const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	const auto ending_chunk_aligned = (ssize_t)(((size_t)ending_chunk >> 1) << 1);
	if (ending_chunk_aligned <= starting_chunk) {
		for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
			if (_selection.data()[i] != 0) {
				return i;
			}
		}
	} else {
		for (ssize_t i = ending_chunk; i >= ending_chunk_aligned; i--) {
			if (_selection.data()[i] != 0) {
				return i;
			}
		}
		const ssize_t starting_chunk_aligned = ((starting_chunk + 3) / 2) * 2;
		for (ssize_t i = ((ssize_t)ending_chunk_aligned) - 2; i >= starting_chunk_aligned; i -= 2) {
			vec = _mm_load_si128((__m128i*)&_selection.data()[i]);
			if (_mm_testz_si128(vec, ones) == 0) {
				uint64_t DECLARE_ALIGN(16) final_sel[2];
				_mm_store_si128((__m128i*)final_sel, vec);
				// If final_sel[0] == 0, it means that the chunk is in one of the last two 32 bits
				// of vec.
				// Otherwise, it is one of the two first.
				uint8_t reg_pos = (final_sel[1] != 0);
				// uint32_t* preg_last2 = (uint32_t*) &final_sel[reg_pos];
				// uint8_t reg_pos_last2 = (preg_last2[1] != 0);
				// return i + (reg_pos<<1) + reg_pos_last2;
				return i + (reg_pos << 1);
			}
		}
		for (ssize_t i = starting_chunk_aligned - 1; i >= starting_chunk; i--) {
			if (_selection.data()[i] != 0) {
				return i;
			}
		}
	}
	return starting_chunk - 1;

#else
	for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
		if (_selection.data()[i] != 0) {
			return i;
		}
	}
	return starting_chunk - 1;
#endif
}

PVRow PVCore::PVSelBitField::find_next_set_bit(const PVRow index, const PVRow size) const
{
	/**
	 * This function can be optimized by testing chunk per chunk and using __builtin_ctzl
	 * but it takes really few ms to compute on 200 Me
	 */
	if (index > size) {
		return PVROW_INVALID_VALUE;
	}

	for (PVRow i = index; i < size; ++i) {
		if (get_line_fast(i))
			return i;
	}

	return PVROW_INVALID_VALUE;
}

PVRow PVCore::PVSelBitField::find_previous_set_bit(const PVRow index, const PVRow size) const
{
	/**
	 * This function can be optimized by testing chunk per chunk and using __builtin_clzl
	 * but it takes really few ms to compute on 200 Me
	 */
	if (index > size) {
		return PVROW_INVALID_VALUE;
	}

	for (PVRow i = index; i > 0; --i) {
		if (get_line_fast(i))
			return i;
	}

	// as unsigned can not be negative, the case 0 is tested outside of the loop
	if (get_line_fast(0))
		return 0;

	return PVROW_INVALID_VALUE;
}

void PVCore::PVSelBitField::serialize_write(PVCore::PVSerializeObject& so) const
{
	size_t mem_size = pvcop::core::__impl::bit_manip::to_mem_size(_selection.size());
	so.buffer_write("selection_data", _selection.data(), mem_size);
	int size = _selection.size();
	so.attribute_write("selection_size", size);
}

PVCore::PVSelBitField PVCore::PVSelBitField::serialize_read(PVCore::PVSerializeObject& so)
{
	int size = so.attribute_read<int>("selection_size");
	size_t mem_size = pvcop::core::__impl::bit_manip::to_mem_size(size);
	PVCore::PVSelBitField sel(size);
	so.buffer_read("selection_data", sel._selection.data(), mem_size);
	return sel;
}
