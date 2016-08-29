/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLinesProperties.h>

/******************************************************************************
 *
 * Inendi::PVLinesProperties::A2B_copy_restricted_by_selection
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::A2B_copy_restricted_by_selection(Inendi::PVLinesProperties& b,
                                                                 PVSelection const& selection) const
{
	selection.visit_selected_lines([&](const PVRow r) { b._colors[r] = this->_colors[r]; });
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::reset_to_default_color()
{
	std::fill(_colors.begin(), _colors.end(), 0xFF); // FIXME : should use PVCore::PVHSVColor::WHITE
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::selection_set_color(PVSelection const& selection,
                                                    const PVCore::PVHSVColor c)
{
	selection.visit_selected_lines([&](const PVRow r) { this->_colors[r] = c; }, selection.count());
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::serialize
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::serialize_write(PVCore::PVSerializeObject& so)
{
	PVRow row_count = _colors.size();
	so.attribute("row_count", row_count);
	so.buffer("lp_data", _colors, row_count);
}

Inendi::PVLinesProperties Inendi::PVLinesProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRow row_count;
	so.attribute("row_count", row_count);

	Inendi::PVLinesProperties lp(row_count);
	so.buffer("lp_data", lp._colors, row_count);

	return lp;
}
