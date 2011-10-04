//! \file PVSelection.cpp
//! $Id: PVSelection.cpp 3215 2011-06-29 06:59:45Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include "bithacks.h"

#include <picviz/PVSelection.h>

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

	size_t tmp = (1 << 5) - 1; // Used for modulus operations (%32)

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
	table.resize(PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	select_none();
}

/******************************************************************************
 *
 * Picviz::PVSelection::PVSelection(std::vector<PVRow>)
 *
 *****************************************************************************/
Picviz::PVSelection::PVSelection(std::vector<PVRow> const& rtable)
{
	table.resize(PICVIZ_SELECTION_NUMBER_OF_CHUNKS);
	select_none();

	std::vector<PVRow>::const_iterator it;
	for (it = rtable.begin(); it != rtable.end(); it++) {
		set_line(*it, true);
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::get_line
 *
 *****************************************************************************/
bool Picviz::PVSelection::get_line(pv_row line_index) const
{
	pv_row pos;
	pv_row shift;

	/*
	 * Say you want to retrieve if the line 20000 is selected or not:
	 * pos = 312
	 * shift = 32
	 */
	pos = line_index / PICVIZ_SELECTION_CHUNK_SIZE;
	shift = line_index - (pos * PICVIZ_SELECTION_CHUNK_SIZE);

	return B_IS_SET(table[pos], shift);
}

/******************************************************************************
 *
 * Picviz::PVSelection::get_number_of_selected_lines_in_range
 *
 *****************************************************************************/
int Picviz::PVSelection::get_number_of_selected_lines_in_range(pv_row a, pv_row b) const
{
	
	pv_row line_index;
	int count = 0; 

	for (line_index = a; line_index<b; line_index++) { 
		if (get_line(line_index)) { 
			count++;
		}
	}

	return count;
	

	// return count_bits_between(a, b-1, &table[0]);
}

std::vector<PVRow> Picviz::PVSelection::get_rows_table()
{
	std::vector<PVRow> rtable;
	rtable.reserve(PICVIZ_LINES_MAX);

	pv_row line_index;

	for (line_index = 0; line_index < PICVIZ_LINES_MAX; line_index++) {
		if (get_line(line_index)) {
			rtable.push_back(line_index);
		}
	}

	return rtable;
}

/******************************************************************************
 *
 * Picviz::PVSelection::operator&
 *
 *****************************************************************************/
Picviz::PVSelection Picviz::PVSelection::operator&(const PVSelection &rhs) const
{
	Picviz::PVSelection result = *this;
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] &= rhs.table[i];
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		result.table[i] = ~table[i];
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] |= rhs.table[i];
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] &= ~rhs.table[i];
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] ^= rhs.table[i];
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
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] = 0xFFFFFFFF;
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_even
 *
 *****************************************************************************/
void Picviz::PVSelection::select_even()
{
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] = 0xAAAAAAAA;
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_none
 *
 *****************************************************************************/
void Picviz::PVSelection::select_none()
{
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] = 0;
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_odd
 *
 *****************************************************************************/
void Picviz::PVSelection::select_odd()
{
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] = 0x55555555;
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::select_inverse
 *
 *****************************************************************************/
void Picviz::PVSelection::select_inverse()
{
	for (pv_row i = 0; i < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; i++) {
		table[i] = ~table[i];
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::set_line
 *
 *****************************************************************************/
void Picviz::PVSelection::set_line(PVRow line_index, bool bool_value)
{
	pv_row pos;
	pv_row shift;

	pos = line_index / PICVIZ_SELECTION_CHUNK_SIZE;
	shift = line_index - (pos * PICVIZ_SELECTION_CHUNK_SIZE);

	if ( bool_value )  {
		B_SET(table[pos], shift);
	} else {
		B_UNSET(table[pos], shift);
	}
}

/******************************************************************************
 *
 * Picviz::PVSelection::get_buffer
 *
 *****************************************************************************/
void* Picviz::PVSelection::get_buffer()
{
	return &table[0];
}

void Picviz::PVSelection::write_selected_lines_nraw(QTextStream& stream, PVRush::PVNraw const& nraw)
{
	PVRow nrows = nraw.table.size();
	assert(nrows > 0);
	PVCol ncols = nraw.table[0].size();
	assert(ncols > 0);
	for (int line_index = 0; line_index < nraw.table.size(); line_index++) {
		if (!get_line(line_index)) {
			continue;
		}

		QString line = nraw.nraw_line_to_csv(line_index);
		stream << line << QString("\n");
	}
}

void Picviz::PVSelection::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.buffer("selection_data", &table.at(0), PICVIZ_SELECTION_NUMBER_OF_BYTES);
}
