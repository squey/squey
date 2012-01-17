//! \file PVLinesProperties.cpp
//! $Id: PVLinesProperties.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <picviz/general.h>

#include <picviz/PVLinesProperties.h>

#include <string.h>		// for memset()

// Picviz::PVLineProperties::PVLineProperties(unsigned char default_color)
// {
// 	color.r() = 255;
// 	color.g() = 255;
// 	color.b() = 255;
// 	color.a() = 255;
// }

/******************************************************************************
 *
 * Picviz::PVLinesProperties::PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::PVLinesProperties()
{
	// We initialize a default color as white and fully opaque
	PVCore::PVColor color = PVCore::PVColor::fromRgba(255, 255, 255, 255);
	
	table.resize(PICVIZ_LINESPROPS_NUMBER_OF_CHUNKS, color);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::~PVLinesProperties
 *
 *****************************************************************************/
Picviz::PVLinesProperties::~PVLinesProperties()
{
	// Shall we delete the table?
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2A_set_to_line_properties_restricted_by_selection_and_nelts(PVCore::PVColor line_properties,  PVSelection const& selection, PVRow nelts)
{
	PVRow row;

	for (row=0; row<nelts; row++) {
		if (selection.get_line(row)) {
			table[row] = line_properties;
		}
	}
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts(Picviz::PVLinesProperties &b,  PVSelection const& selection, PVRow nelts)
{
	PVRow row;

	for (row=0; row < nelts; row++) {
		if (selection.get_line(row)) {
			b.table[row] = table[row];
		}
	}
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
	}
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::get_line_properties
 *
 *****************************************************************************/
PVCore::PVColor& Picviz::PVLinesProperties::get_line_properties(PVRow line)
{
	return table[line];
}

const PVCore::PVColor& Picviz::PVLinesProperties::get_line_properties(PVRow line) const
{
	return table[line];
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_get_a
 *
 *****************************************************************************/
unsigned char Picviz::PVLinesProperties::line_get_a(PVRow line)
{
	return table[line].a();
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_get_r
 *
 *****************************************************************************/
unsigned char Picviz::PVLinesProperties::line_get_r(PVRow line)
{
	return table[line].r();
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_get_b
 *
 *****************************************************************************/
unsigned char Picviz::PVLinesProperties::line_get_b(PVRow line)
{
	return table[line].b();
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_get_g
 *
 *****************************************************************************/
unsigned char Picviz::PVLinesProperties::line_get_g(PVRow line)
{
	return table[line].g();
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_a
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_a(PVRow line, unsigned char a)
{
	table[line].a() = a;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_r
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_r(PVRow line, unsigned char r)
{
	table[line].r() = r;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_b
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_b(PVRow line, unsigned char b)
{
	table[line].b() = b;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_g
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_g(PVRow line, unsigned char g)
{
	table[line].g() = g;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_rgb
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_rgb(PVRow line, unsigned char r, unsigned char g, unsigned char b)
{
	line_set_r(line, r);
	line_set_g(line, g);
	line_set_b(line, b);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_rgba
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_rgba(PVRow line, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	line_set_r(line, r);
	line_set_g(line, g);
	line_set_b(line, b);
	line_set_a(line, a);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_rgb_from_color
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_rgb_from_color(PVRow line, PVCore::PVColor color)
{
	unsigned char a;
	a = line_get_a(line);
	table[line] = color;
	line_set_a(line, a);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::line_set_rgba_from_color
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::line_set_rgba_from_color(PVRow line, const PVCore::PVColor &color)
{
	table[line] = color;
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

	table = rhs.table;
	
	return *this;
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::reset_to_default_color()
{
	memset(&table[0], 0xFF, PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
}

/******************************************************************************
 *
 * Picviz::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Picviz::PVLinesProperties::selection_set_rgba(PVSelection const& selection, PVRow nelts, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	PVRow row;
	
	for (row=0; row < nelts; row++) {
		if (selection.get_line(row)) {
			line_set_rgba(row, r, g, b, a);
		}
	}
}

void Picviz::PVLinesProperties::debug()
{
	PVRow row;

	for (row=0; row<table.size(); row++) {
		PVCore::PVColor &c = table[row];
		PVLOG_INFO("%d: %d %d %d\n", row, c.r(), c.g(), c.b());
	}
}

void Picviz::PVLinesProperties::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.buffer("lp_data", &table[0], PICVIZ_LINESPROPS_NUMBER_OF_BYTES);
}
