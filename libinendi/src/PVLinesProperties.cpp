/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLinesProperties.h>

/******************************************************************************
 *
 * Inendi::PVLinesProperties::ensure_allocated
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::ensure_allocated(size_t row_count)
{
	if (_colors.size() == 0) {
		_colors.resize(row_count);
	}
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::ensure_initialized
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::ensure_initialized(size_t row_count)
{
	if (_colors.size() == 0) {
		_colors.resize(row_count);
		reset_to_default_color(row_count);
	}
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::A2B_copy_restricted_by_selection_and_nelts(
    Inendi::PVLinesProperties& b, PVSelection const& selection, PVRow nelts) const
{
	if (not _colors.size()) {
		return;
	}

	if (not b._colors.size()) {
		b.ensure_initialized(nelts);
	}

	selection.visit_selected_lines([&](const PVRow r) { b._colors[r] = this->_colors[r]; }, nelts);
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::reset_to_default_color(PVRow row_count)
{
	ensure_allocated(row_count);
	std::fill(_colors.begin(), _colors.end(), 0xFF); // FIXME : should use PVCore::PVHSVColor::WHITE
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::selection_set_color(PVSelection const& selection,
                                                    const PVRow nelts,
                                                    const PVCore::PVHSVColor c)
{
	ensure_initialized(nelts);

	selection.visit_selected_lines([&](const PVRow r) { this->_colors[r] = c; }, nelts);
}

/******************************************************************************
 *
 * Inendi::PVLinesProperties::serialize
 *
 *****************************************************************************/
void Inendi::PVLinesProperties::serialize(PVCore::PVSerializeObject& so,
                                          PVCore::PVSerializeArchive::version_t /*v*/)
{
	PVRow row_count = _colors.size();
	so.attribute("row_count", row_count);
	so.buffer("lp_data", _colors, row_count);
}
