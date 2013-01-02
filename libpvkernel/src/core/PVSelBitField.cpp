/**
 * \file PVSelBitField.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/rush/PVNraw.h>

/******************************************************************************
 *
 * PVCore::PVSelBitField::PVSelBitField()
 *
 *****************************************************************************/
PVCore::PVSelBitField::PVSelBitField():
	_table(nullptr)
{
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::PVSelBitField(std::vector<PVRow>)
 *
 *****************************************************************************/
PVCore::PVSelBitField::PVSelBitField(std::vector<PVRow> const& r_table):
	_table(nullptr)
{
	allocate_table();
	select_none();

	std::vector<PVRow>::const_iterator it;
	for (it = r_table.begin(); it != r_table.end(); it++) {
		set_line(*it, true);
	}
}

PVCore::PVSelBitField::PVSelBitField(PVSelBitField const& o):
	_table(nullptr)
{
	allocate_and_copy_from(o);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::get_number_of_selected_lines_in_range
 *
 *****************************************************************************/
size_t PVCore::PVSelBitField::get_number_of_selected_lines_in_range(PVRow a, PVRow b) const
{
	if (!_table) {
		return 0;
	}
#if 0	
	PVRow line_index;
	size_t count = 0;

	for (line_index = a; line_index<b; line_index++) { 
		if (get_line(line_index)) { 
			count++;
		}
	}

	return count;
#endif
	assert(b > a);
	
	return PVCore::PVBitCount::bit_count_between(a, b-1, &_table[0]);
}

std::vector<PVRow> PVCore::PVSelBitField::get_rows_table()
{
	if (!_table) {
		return std::vector<PVRow>();
	}

	std::vector<PVRow> r_table;
	r_table.reserve(PICVIZ_LINES_MAX);

	PVRow line_index;

	for (line_index = 0; line_index < PICVIZ_LINES_MAX; line_index++) {
		if (get_line(line_index)) {
			r_table.push_back(line_index);
		}
	}

	return r_table;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::is_empty
 *
 *****************************************************************************/
bool PVCore::PVSelBitField::is_empty() const
{
	if (!_table) {
		return true;
	}
#ifdef __SSE4_1__
	const __m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i += 4) {
		vec = _mm_load_si128((__m128i*) &_table[i]);
		if (_mm_testz_si128(vec, ones) == 0) {
			return false;
		}
	}
#if (PICVIZ_SELECTION_NUMBER_OF_CHUNKS % 4 != 0)
	for (PVRow i = (PICVIZ_SELECTION_NUMBER_OF_CHUNKS/4)*4; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		if (_table[i] != 0) {
			return false;
		}
	}
#endif
	return true;

#else
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		if (_table[i] != 0) {
			return false;
		}
	}
	return true;
#endif
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::is_empty_between
 *
 *****************************************************************************/
bool PVCore::PVSelBitField::is_empty_between(PVRow const a, PVRow b) const
{
	assert(b > a);
	assert(b <= PICVIZ_LINES_MAX);

	if (!_table) {
		return true;
	}
	
	// 'b' is not included
	b--;

	size_t chunk_start = a/PICVIZ_SELECTION_CHUNK_SIZE;
	const size_t chunk_end   = b/PICVIZ_SELECTION_CHUNK_SIZE;
	
	const PVRow cbit = line_index_to_chunk_bit(a);
	if (cbit > 0) {
		// Prelogue
		uint64_t cv = _table[chunk_start];
		if (chunk_end == chunk_start) {
			const size_t off = PICVIZ_SELECTION_CHUNK_SIZE-line_index_to_chunk_bit(b)-1;
			cv = (cv << off) >> off;
			return (cv >> cbit) == 0;
		}
		if ((cv >> cbit) != 0) {
			return false;
		}
		chunk_start++;
	}

	// Main loop
	for (size_t c = chunk_start; c < chunk_end; c++) {
		if (_table[c] != 0) {
			return false;
		}
	}

	// Epilogue
	return (_table[chunk_end] << (PICVIZ_SELECTION_CHUNK_SIZE-line_index_to_chunk_bit(b)-1)) == 0;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator=
 *
 *****************************************************************************/
PVCore::PVSelBitField& PVCore::PVSelBitField::operator=(const PVSelBitField &rhs)
{
	if (&rhs == this) {
		return *this;
	}

	if (rhs._table) {
		if (!_table) {
			allocate_table();
		}
		copy_from(rhs);
	}
	else {
		if (_table) {
			memset(_table, 0x00, PICVIZ_SELECTION_NUMBER_OF_BYTES);
		}
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator&(const PVSelBitField &rhs) const
{
	PVCore::PVSelBitField result(*this);
	result &= rhs;

	return std::move(result);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator&=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator&=(const PVSelBitField &rhs)
{
	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= rhs._table[i];
	}

	return *this;
}

PVCore::PVSelBitField& PVCore::PVSelBitField::and_optimized(const PVSelBitField& rhs)
{
	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	const ssize_t last_chunk = get_max_last_nonzero_chunk_index(rhs);
	if (last_chunk >= 0) {
		for (PVRow i = 0; i < (PVRow) last_chunk; i++) {
			_table[i] &= rhs._table[i];
		}
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator~
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator~() const
{
	PVSelBitField result;
	result.allocate_table();
	if (!_table) {
		result.select_all();
	}
	else {
		for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
			result._table[i] = ~_table[i];
		}
	}

	return std::move(result);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator|
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator|(const PVSelBitField &rhs) const
{
	PVSelBitField result(*this);
	result |= rhs;

	return std::move(result);
}

/******************************************************************************
*
 * PVCore::PVSelBitField::operator|=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator|=(const PVSelBitField &rhs)
{ 
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}


	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] |= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator-(const PVSelBitField &rhs) const
{
	PVSelBitField result = *this;
	result -= rhs;

	return std::move(result);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator-=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator-=(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= ~rhs._table[i];
	}

	return *this;
}

void PVCore::PVSelBitField::AB_sub(PVSelBitField const& a, PVSelBitField const& b)
{
	static size_t nthreads = PVCore::PVHardwareConcurrency::get_physical_core_number();

	if (!_table) {
		allocate_table();
	}

	if (!a._table) {
		select_none();
		return;
	}

	if (!b._table) {
		select_all();
	}

#pragma omp parallel for num_threads(nthreads)
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] = a._table[i] & (~b._table[i]);
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^
 *
 *****************************************************************************/
PVCore::PVSelBitField PVCore::PVSelBitField::operator^(const PVSelBitField &rhs) const
{
	PVSelBitField result = *this;
	result ^= rhs;

	return std::move(result);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::operator^=
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::operator^=(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] ^= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::or_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::or_not(const PVSelBitField &rhs)
{
	if ((&rhs == this)) {
		return *this;
	}

	if (!rhs._table) {
		select_all();
		return *this;
	}

	if (!_table) {
		allocate_table();
		for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
			_table[i] = ~(rhs._table[i]);
		}
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] |= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::and_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::and_not(const PVSelBitField &rhs)
{
	if ((&rhs == this) || (rhs._table == NULL)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		select_none();
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::xor_not
 *
 *****************************************************************************/
PVCore::PVSelBitField & PVCore::PVSelBitField::xor_not(const PVSelBitField &rhs)
{
	if (&rhs == this) {
		return *this;
	}

	if (!rhs._table) {
		select_all();
		return *this;
	}

	if (!_table) {
		allocate_table();
		for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
			_table[i] = ~(rhs._table[i]);
		}
		return *this;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] ^= ~(rhs._table[i]);
	}

	return *this;
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_all
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_all()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0xFF, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

void PVCore::PVSelBitField::select_random()
{
	if (!_table) {
		allocate_table();
	}
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] = (uint64_t) (rand()) | ((uint64_t)(rand()) << 32);
	}
}



/******************************************************************************
 *
 * PVCore::PVSelBitField::select_even
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_even()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0xAA, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_none
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_none()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0x00, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_odd
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_odd()
{
	if (!_table) {
		allocate_table();
	}
	memset(_table, 0x55, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::select_inverse
 *
 *****************************************************************************/
void PVCore::PVSelBitField::select_inverse()
{
	if (!_table) {
		allocate_table();
		select_all();
		return;
	}

	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] = ~_table[i];
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::set_line
 *
 *****************************************************************************/
void PVCore::PVSelBitField::set_line(PVRow line_index, bool bool_value)
{
	if (!_table) {
		allocate_table();
		select_none();
	}

	const PVRow pos = line_index_to_chunk(line_index);
	const PVRow shift = line_index_to_chunk_bit(line_index);
	
	if ( bool_value )  {
		_table[pos] |= 1UL<<shift;
	} else {
		_table[pos] &= ~(1UL<<shift);
	}
}

/******************************************************************************
 *
 * PVCore::PVSelBitField::set_line_select_only
 *
 *****************************************************************************/
void PVCore::PVSelBitField::set_line_select_only(PVRow line_index, bool bool_value)
{
	if (bool_value == false) {
		return;
	}

	set_line(line_index, true);
}

ssize_t PVCore::PVSelBitField::get_last_nonzero_chunk_index(ssize_t starting_chunk, ssize_t ending_chunk) const
{
	if (!_table || (starting_chunk < 0 || ending_chunk < 0)) {
		return -1;
	}
#ifdef __SSE4_1__
	__m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	const ssize_t ending_chunk_aligned = (ssize_t)(((size_t)ending_chunk>>1)<<1);
	if (ending_chunk_aligned <= starting_chunk) {
		for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
			if (_table[i] != 0) {
				return i;
			}
		}
	}
	else {
		for (ssize_t i = ending_chunk; i >= ending_chunk_aligned; i--) {
			if (_table[i] != 0) {
				return i;
			}
		}
		for (ssize_t i = ((ssize_t)ending_chunk_aligned)-2; i >= starting_chunk; i -= 2) {
			vec = _mm_load_si128((__m128i*) &_table[i]);
			if (_mm_testz_si128(vec, ones) == 0) {
				uint64_t DECLARE_ALIGN(16) final_sel[2];
				_mm_store_si128((__m128i*)final_sel, vec);
				// If final_sel[0] == 0, it means that the chunk is in one of the last two 32 bits of vec.
				// Otherwise, it is one of the two first.
				uint8_t reg_pos = (final_sel[1] != 0);
				//uint32_t* preg_last2 = (uint32_t*) &final_sel[reg_pos];
				//uint8_t reg_pos_last2 = (preg_last2[1] != 0);
				//return i + (reg_pos<<1) + reg_pos_last2;
				return i + (reg_pos<<1);
			}
		}
	}
	return starting_chunk-1;

#else
	for (ssize_t i = ending_chunk; i >= starting_chunk; i--) {
		if (_table[i] != 0) {
			return i;
		}
	}
	return starting_chunk-1;
#endif
}

PVCore::PVSelBitField& PVCore::PVSelBitField::or_optimized(const PVSelBitField& rhs)
{
	if ((&rhs == this) || (!rhs._table)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	const ssize_t last_chunk = rhs.get_last_nonzero_chunk_index();
	if (last_chunk >= 0) {
		// TODO: GCC vectorize this, but we could try to
		// do this by SSE and checking whether rhs._table[] vectors
		// are non-null before doing the actual OR+store (thus saving stores)
		for (PVRow i = 0; i <= (PVRow) last_chunk; i++) {
			_table[i] |= rhs._table[i];
		}
	}

	return *this;
}

PVCore::PVSelBitField& PVCore::PVSelBitField::or_range(const PVSelBitField &rhs,
                                                       PVRow start, PVRow end)
{
	if ((&rhs == this) || (!rhs._table)) {
		return *this;
	}

	if (!_table) {
		allocate_table();
		copy_from(rhs);
		return *this;
	}

	PVRow chunk_start = line_index_to_chunk(start);
	PVRow chunk_end = line_index_to_chunk(end);

	for (PVRow i = chunk_start; i <= chunk_end; ++i) {
		_table[i] |= rhs._table[i];
	}

	return *this;
}


ssize_t PVCore::PVSelBitField::get_min_last_nonzero_chunk_index(PVSelBitField const& other) const
{
	if (!_table) {
		return -1;
	}
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(0, last_chunk);
}

ssize_t PVCore::PVSelBitField::get_max_last_nonzero_chunk_index(PVSelBitField const& other) const
{
	if (!_table) {
		return -1;
	}
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(last_chunk, PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1);
}

void PVCore::PVSelBitField::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (so.is_writing()) {
		if (_table) {
			so.buffer("selection_data", _table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
		}
	}
	else {
		if (so.buffer_exists("selection_data")) {
			if (!_table) {
				allocate_table();
			}
			so.buffer("selection_data", &_table[0], PICVIZ_SELECTION_NUMBER_OF_BYTES);
		}
		else {
			if (_table) {
				select_none();
			}
		}
	}
}
