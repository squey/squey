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
Picviz::PVLinesProperties::PVLinesProperties():
	_table(nullptr)
{
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::PVLinesProperties(const PVLinesProperties & rhs):
	_table(nullptr)
{
	if (rhs._table) {
		allocate_table();
		memcpy(_table, rhs._table, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
	}
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::~PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::~PVLinesProperties()
{
	if (_table) {
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
	if (!_table) {
		reset_to_default_color();
	}

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
	if (!_table) {
		return;
	}

	if (!b._table) {
		b.reset_to_default_color();
	}

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
 * Picviz::PVLinesProperties::operator=
 *
 *****************************************************************************/
Picviz::PVLinesProperties & Picviz::PVLinesProperties::operator=(const PVLinesProperties & rhs)
{
	// We check for self assignment
	if (this == &rhs) {
		return *this;
	}

	if (!rhs._table) {
		if (_table) {
			reset_to_default_color();
		}
	}
	else {
		if (!_table) {
			allocate_table();
		}
		memcpy(_table, rhs._table, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
	}

	return *this;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::reset_to_default_color()
{
	if (!_table) {
		allocate_table();
	}
	memset(&_table[0], 0xFF, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::selection_set_color(PVSelection const& selection, const PVRow nelts, const PVCore::PVHSVColor c)
{
	if (!_table) {
		reset_to_default_color();
	}
	selection.visit_selected_lines([&](const PVRow r) {
		this->line_set_color(r, c);
	},
	nelts);
}

void Picviz::PVLinesProperties::set_random(const PVRow n)
{
	if (!_table) {
		allocate_table();
	}
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor(rand() % ((1<<HSV_COLOR_NBITS_ZONE)*6)));
	}
}

void Picviz::PVLinesProperties::set_linear(const PVRow n)
{
	if (!_table) {
		allocate_table();
	}
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6)-1;
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor((uint8_t)(((double)(i*color_max)/(double)n)*color_max)));
	}
}

void Picviz::PVLinesProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (so.is_writing()) {
		if (_table) {
			so.buffer("lp_data", &_table[0], PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
		}
	}
	else {
		if (so.buffer_exists("lp_data")) {
			if (!_table) {
				allocate_table();
			}
			so.buffer("lp_data", &_table[0], PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
		}
		else {
			if (_table) {
				reset_to_default_color();
			}
		}
	}
}
