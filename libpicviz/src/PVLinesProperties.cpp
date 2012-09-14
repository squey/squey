/**
 * \file PVLinesProperties.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <picviz/general.h>

#include <picviz/PVLinesProperties.h>

#include <stdlib.h>     // for rand()
#include <string.h>		// for memset()

Picviz::PVLinesProperties::color_allocator_type Picviz::PVLinesProperties::_color_allocator;

/******************************************************************************
 *
 * Picviz::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::PVLinesProperties()
{
	_table = _color_allocator.allocate(PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS);
	reset_to_default_color();
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::PVLinesProperties(const PVLinesProperties & rhs)
{
	_table = _color_allocator.allocate(PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS);
	std::copy(rhs._table, rhs._table + PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS, _table);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::~PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::~PVLinesProperties()
{
	if(_table != 0) {
		_color_allocator.deallocate(_table, PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS);
	}
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVHSVColor line_properties,  PVSelection const& selection, PVRow nelts)
{
	selection.visit_selected_lines([&](const PVRow r) {
			this->_table[r] = line_properties;
		}, nelts);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts(Picviz::PVLinesProperties &b,  PVSelection const& selection, PVRow nelts)
{
	selection.visit_selected_lines([&](const PVRow r) {
			b._table[r] = this->_table[r];
		}, nelts);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2B_copy_zombie_off_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2B_copy_zombie_off_restricted_by_selection_and_nelts(Picviz::PVLinesProperties &b,  PVSelection const& selection, PVRow nelts)
{
	A2B_copy_restricted_by_selection_and_nelts(b, selection, nelts); // FIXME, this is the same code than the previous function, should remove it to make it more generic
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2B_copy_zombie_on_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2B_copy_zombie_on_restricted_by_selection_and_nelts(Picviz::PVLinesProperties &b, PVSelection const& selection, PVRow nelts)
{
	/*
	PVRow row;

	for (row=0; row < nelts; row++) {
		if (selection.get_line(row)) {
			b.table[row] = table[row];
		} else {
			b.table[row].a() = line_get_a(row)/2;
			b.table[row].b() = line_get_b(row)/2;
			b.table[row].g() = line_get_g(row)/2;
			b.table[row].r() = line_get_r(row)/2;
		}
	}*/
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::operator=
 *
 *****************************************************************************/
Picviz::PVLinesProperties & Picviz::PVLinesProperties::operator=(const PVLinesProperties & rhs)
{
	// We check for self assignment
	if (this == &rhs) {
		return *this;
	}

	std::copy(rhs._table, rhs._table + PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS, _table);

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::reset_to_default_color()
{
	memset(&_table[0], 0xFF, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::selection_set_color(PVSelection const& selection, const PVRow nelts, const PVCore::PVHSVColor c)
{
	selection.visit_selected_lines([&](const PVRow r) {
		this->line_set_color(r, c);
	},
	nelts);
}

void Picviz::PVLinesProperties::set_random(const PVRow n)
{
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor(rand() % ((1<<HSV_COLOR_NBITS_ZONE)*6)));
	}
}

void Picviz::PVLinesProperties::set_linear(const PVRow n)
{
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6)-1;
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor((uint8_t)(((double)(i*color_max)/(double)n)*color_max)));
	}
}

void Picviz::PVLinesProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.buffer("lp_data", &_table[0], PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
}
