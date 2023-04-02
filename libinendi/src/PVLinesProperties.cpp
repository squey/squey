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

#include <inendi/PVLinesProperties.h>
#include <inendi/PVSelection.h>

#include <pvkernel/core/PVHSVColor.h> // for PVHSVColor
#include <pvkernel/core/PVSerializeObject.h>

#include <pvbase/types.h> // for PVRow

#include <algorithm> // for fill

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
	std::fill(_colors.begin(), _colors.end(),
	          PVCore::PVHSVColor(0xFF)); // FIXME : should use PVCore::PVHSVColor::WHITE
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
void Inendi::PVLinesProperties::serialize_write(PVCore::PVSerializeObject& so) const
{
	PVRow row_count = _colors.size();
	so.attribute_write("row_count", row_count);
	so.buffer_write("lp_data", _colors);
}

Inendi::PVLinesProperties Inendi::PVLinesProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	auto row_count = so.attribute_read<PVRow>("row_count");

	Inendi::PVLinesProperties lp(row_count);
	so.buffer_read("lp_data", lp._colors, row_count);

	return lp;
}
