/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/general.h>

#include <inendi/PVLinesProperties.h>

#include <stdlib.h>     // for rand()
#include <string.h>		// for memset()

Inendi::PVLinesProperties::color_allocator_type Inendi::PVLinesProperties::_color_allocator;

/******************************************************************************
 *
 * Inendi::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Inendi::PVLinesProperties::PVLinesProperties():
	_table(nullptr)
{
	allocate_table();
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Inendi::PVLinesProperties::PVLinesProperties(const PVLinesProperties & rhs):
	_table(nullptr)
{
	if (rhs._table) {
		allocate_table();
		memcpy(_table, rhs._table, INENDI_LINESPROPS_NUMBER_OF_BYTES);
	}
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::~PVLinesProperties
 *
 *****************************************************************************/
Inendi::PVLinesProperties::~PVLinesProperties()
{
	if (_table) {
		_color_allocator.deallocate(_table, INENDI_LINESPROPS_NUMBER_OF_CHUNKS);
	}
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVHSVColor line_properties,  PVSelection const& selection, PVRow nelts)
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
 * Inendi::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts(Inendi::PVLinesProperties &b,  PVSelection const& selection, PVRow nelts)
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
 * Inendi::PVLinesProperties::operator=
 *
 *****************************************************************************/
Inendi::PVLinesProperties & Inendi::PVLinesProperties::operator=(const PVLinesProperties & rhs)
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
		memcpy(_table, rhs._table, INENDI_LINESPROPS_NUMBER_OF_BYTES);
	}

	return *this;
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::reset_to_default_color()
{
	if (!_table) {
		allocate_table();
	}
	memset(&_table[0], 0xFF, INENDI_LINESPROPS_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::selection_set_color(PVSelection const& selection, const PVRow nelts, const PVCore::PVHSVColor c)
{
	if (!_table) {
		reset_to_default_color();
	}
	selection.visit_selected_lines([&](const PVRow r) {
		this->line_set_color(r, c);
	},
	nelts);
}

void Inendi::PVLinesProperties::set_random(const PVRow n)
{
	if (!_table) {
		allocate_table();
	}
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor(rand() % ((1<<HSV_COLOR_NBITS_ZONE)*6)));
	}
}

void Inendi::PVLinesProperties::set_linear(const PVRow n)
{
	if (!_table) {
		allocate_table();
	}
	constexpr static size_t color_max = ((1<<HSV_COLOR_NBITS_ZONE)*6)-1;
	for (PVRow i = 0; i < n; i++) {
		line_set_color(i, PVCore::PVHSVColor((uint8_t)(((double)(i*color_max)/(double)n)*color_max)));
	}
}

void Inendi::PVLinesProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	if (so.is_writing()) {
		if (_table) {
			so.buffer("lp_data", &_table[0], INENDI_LINESPROPS_NUMBER_OF_BYTES);
		}
	}
	else {
		if (so.buffer_exists("lp_data")) {
			if (!_table) {
				allocate_table();
			}
			so.buffer("lp_data", &_table[0], INENDI_LINESPROPS_NUMBER_OF_BYTES);
		}
		else {
			if (_table) {
				reset_to_default_color();
			}
		}
	}
}
