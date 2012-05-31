//! \file PVSelection.cpp
//! $Id: PVSelection.cpp 3215 2011-06-29 06:59:45Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include "bithacks.h"

#include <pvkernel/core/picviz_intrin.h>

#include <picviz/PVSelection.h>
#include <picviz/PVSparseSelection.h>

static inline uint32_t count_bits(size_t n, const uint32_t* data)
{
	uint32_t ret = 0;
	for (size_t i = 0; i < n; i++) {
		uint32_t v = data[i];
		COUNT_BITS_UINT32(v,ret);
	}
	return ret;
}

// a and b are positions in bits and are inclusive (which means that b-a+1 bits are checked)
// No boundary checks are done, so be carefull !!
static uint32_t count_bits_between(size_t a, size_t b, const uint32_t* data)
{
	size_t a_byte = a >> 5; // = a/32
	size_t b_byte = b >> 5;

	const size_t tmp = (1 << 5) - 1; // Used for modulus operations (%32)

	if (a_byte == b_byte) {
		uint8_t shift_left = a & tmp;
		uint32_t v = data[a_byte] << shift_left;
		v >>= shift_left + (32U - (b & tmp) - 1);
		uint32_t ret = 0;
		COUNT_BITS_UINT32(v,ret);
		return ret;
	}

	// Hard part is done here
	uint32_t ret = count_bits(b_byte - a_byte - 1, data + a_byte + 1);

	// Finish it
	uint32_t v = data[a_byte] << (a & tmp); // a & tmp = a%32
	COUNT_BITS_UINT32(v,ret);

	v = data[b_byte] >> (32U - (b & tmp) - 1);
	COUNT_BITS_UINT32(v,ret);

	return ret;
}

/******************************************************************************
 *
 * Picviz::PVSelection::PVSelection()
 *
 *****************************************************************************/
Picviz::PVSelection::PVSelection()
{
	allocate_table();
	select_none();
}

/******************************************************************************
 *
 * Picviz::PVSelection::PVSelection(std::vector<PVRow>)
 *
 *****************************************************************************/
Picviz::PVSelection::PVSelection(std::vector<PVRow> const& r_table)
{
	allocate_table();
	select_none();

	std::vector<PVRow>::const_iterator it;
	for (it = r_table.begin(); it != r_table.end(); it++) {
		set_line(*it, true);
	}
}

Picviz::PVSelection::PVSelection(PVSelection const& o)
{
	allocate_and_copy_from(o);
}

/******************************************************************************
 *
 * Picviz::PVSelection::get_number_of_selected_lines_in_range
 *
 *****************************************************************************/
int Picviz::PVSelection::get_number_of_selected_lines_in_range(PVRow a, PVRow b) const
{
	
	PVRow line_index;
	int count = 0; 

	for (line_index = a; line_index<b; line_index++) { 
		if (get_line(line_index)) { 
			count++;
		}
	}

	return count;
	

	// return count_bits_between(a, b-1, &_table[0]);
}

std::vector<PVRow> Picviz::PVSelection::get_rows_table()
{
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
 * Picviz::PVSelection::is_empty
 *
 *****************************************************************************/
bool Picviz::PVSelection::is_empty() const
{
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
 * Picviz::PVSelection::operator=
 *
 *****************************************************************************/
Picviz::PVSelection& Picviz::PVSelection::operator=(const PVSelection &rhs)
{
	if (&rhs == this) {
		return *this;
	}

	memcpy(_table, rhs._table, PICVIZ_SELECTION_NUMBER_OF_CHUNKS);

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator&
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator&(const PVSelection &rhs) const
{
	Picviz::PVSelection result(*this);
	result &= rhs;

	return result;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator&=
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::operator&=(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= rhs._table[i];
	}

	return *this;
}

Picviz::PVSelection& Picviz::PVSelection::and_optimized(const PVSelection& rhs)
{
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
 * Picviz::PVSelection::operator~
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator~() const
{
	PVSelection result;
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		result._table[i] = ~_table[i];
	}

	return result;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator|
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator|(const PVSelection &rhs) const
{
	PVSelection result = *this;
	result |= rhs;

	return result;
}

/******************************************************************************
*
 * Picviz::PVSelection::operator|=
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::operator|=(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] |= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator-
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator-(const PVSelection &rhs) const
{
	PVSelection result = *this;
	result -= rhs;

	return result;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator-=
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::operator-=(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= ~rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator^
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator^(const PVSelection &rhs) const
{
	PVSelection result = *this;
	result ^= rhs;

	return result;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator^=
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::operator^=(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] ^= rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::or_not
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::or_not(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] |= ~rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::and_not
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::and_not(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] &= ~rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::xor_not
 *
 *****************************************************************************/
Picviz::PVSelection & Picviz::PVSelection::xor_not(const PVSelection &rhs)
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] ^= ~rhs._table[i];
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_all
 *
 *****************************************************************************/
void Picviz::PVSelection::select_all()
{
	memset(_table, 0xFF, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_even
 *
 *****************************************************************************/
void Picviz::PVSelection::select_even()
{
	memset(_table, 0xAA, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_none
 *
 *****************************************************************************/
void Picviz::PVSelection::select_none()
{
	memset(_table, 0x00, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_odd
 *
 *****************************************************************************/
void Picviz::PVSelection::select_odd()
{
	memset(_table, 0x55, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_inverse
 *
 *****************************************************************************/
void Picviz::PVSelection::select_inverse()
{
	for (PVRow i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		_table[i] = ~_table[i];
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::set_line
 *
 *****************************************************************************/
void Picviz::PVSelection::set_line(PVRow line_index, bool bool_value)
{
	const PVRow pos = line_index / PICVIZ_SELECTION_CHUNK_SIZE;
	const PVRow shift = line_index - (pos * PICVIZ_SELECTION_CHUNK_SIZE);
	
	if ( bool_value )  {
		B_SET(_table[pos], shift);
	} else {
		B_UNSET(_table[pos], shift);
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::set_line_select_only
 *
 *****************************************************************************/
void Picviz::PVSelection::set_line_select_only(PVRow line_index, bool bool_value)
{
	if (bool_value == false) {
		return;
	}

	set_line(line_index, true);
}

ssize_t Picviz::PVSelection::get_last_nonzero_chunk_index(ssize_t starting_chunk, ssize_t ending_chunk) const
{
	if (starting_chunk < 0 || ending_chunk < 0) {
		return -1;
	}
#ifdef __SSE4_1__
	__m128i ones = _mm_set1_epi32(0xFFFFFFFF);
	__m128i vec;
	const ssize_t ending_chunk_aligned = (ssize_t)(((size_t)ending_chunk>>2)<<2);
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
		for (ssize_t i = ((ssize_t)ending_chunk_aligned)-4; i >= starting_chunk; i -= 4) {
			vec = _mm_load_si128((__m128i*) &_table[i]);
			if (_mm_testz_si128(vec, ones) == 0) {
				uint64_t DECLARE_ALIGN(16) final_sel[2];
				_mm_store_si128((__m128i*)final_sel, vec);
				// If final_sel[0] == 0, it means that the chunk is in one of the last two 32 bits of vec.
				// Otherwise, it is one of the two first.
				uint8_t reg_pos = (final_sel[1] != 0);
				uint32_t* preg_last2 = (uint32_t*) &final_sel[reg_pos];
				uint8_t reg_pos_last2 = (preg_last2[1] != 0);
				return i + (reg_pos<<1) + reg_pos_last2;
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

void Picviz::PVSelection::write_selected_lines_nraw(QTextStream& stream, PVRush::PVNraw const& nraw, PVRow write_max)
{
	PVRow nrows = nraw.get_number_rows();
	assert(nrows > 0);
#ifndef NDEBUG
	PVCol ncols = nraw.get_number_cols();
	assert(ncols > 0);
#endif

	PVRow nrows_counter = 0;

	for (PVRow line_index = 0; line_index < nrows; line_index++) {
		if (!get_line(line_index)) {
			continue;
		}

		nrows_counter++;
		if ((nrows_counter < write_max) || (!write_max)) {
			QString line = nraw.nraw_line_to_csv(line_index);
			stream << line << QString("\n");
		}
	}
}

Picviz::PVSelection& Picviz::PVSelection::or_optimized(const PVSelection& rhs)
{
	const ssize_t last_chunk = rhs.get_last_nonzero_chunk_index();
	if (last_chunk >= 0) {
		// TODO: GCC vectorize this, but we could try to
		// do this by SSE and checking whether rhs._table[] vectors
		// are non-null before doing the actual OR+store (thus saving stores)
		for (PVRow i = 0; i < (PVRow) last_chunk; i++) {
			_table[i] |= rhs._table[i];
		}
	}

	return *this;
}

ssize_t Picviz::PVSelection::get_min_last_nonzero_chunk_index(PVSelection const& other) const
{
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(0, last_chunk);
}

ssize_t Picviz::PVSelection::get_max_last_nonzero_chunk_index(PVSelection const& other) const
{
	const ssize_t last_chunk = other.get_last_nonzero_chunk_index();
	return get_last_nonzero_chunk_index(last_chunk, PICVIZ_SELECTION_NUMBER_OF_CHUNKS-1);
}

void Picviz::PVSelection::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.buffer("selection_data", _table, PICVIZ_SELECTION_NUMBER_OF_BYTES);
}
