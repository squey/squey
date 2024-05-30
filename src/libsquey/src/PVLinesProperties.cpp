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

#include <squey/PVLinesProperties.h>
#include <squey/PVSelection.h>

#include <pvkernel/core/PVHSVColor.h> // for PVHSVColor
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVTheme.h>

#include <pvbase/types.h> // for PVRow

#include <algorithm> // for fill

/******************************************************************************
 *
 * Squey::PVLinesProperties::A2B_copy_restricted_by_selection
 *
 *****************************************************************************/
void Squey::PVLinesProperties::A2B_copy_restricted_by_selection(Squey::PVLinesProperties& b,
                                                                 PVSelection const& selection) const
{
	selection.visit_selected_lines([&](const PVRow r) { b._colors[r] = this->_colors[r]; });
}

/******************************************************************************
 *
 * Squey::PVLinesProperties::reset_to_default_color
 *
 *****************************************************************************/
void Squey::PVLinesProperties::reset_to_default_color()
{
	std::fill(_colors.begin(), _colors.end(), HSV_COLOR_WHITE);
}

/******************************************************************************
 *
 * Squey::PVLinesProperties::selection_set_rgba
 *
 *****************************************************************************/
void Squey::PVLinesProperties::selection_set_color(PVSelection const& selection,
                                                    const PVCore::PVHSVColor c)
{
	selection.visit_selected_lines([&](const PVRow r) { this->_colors[r] = c; }, selection.count());
}

/******************************************************************************
 *
 * Squey::PVLinesProperties::serialize
 *
 *****************************************************************************/
void Squey::PVLinesProperties::serialize_write(PVCore::PVSerializeObject& so) const
{
	PVRow row_count = _colors.size();
	so.attribute_write("row_count", row_count);
	so.buffer_write("lp_data", _colors);
}

Squey::PVLinesProperties Squey::PVLinesProperties::serialize_read(PVCore::PVSerializeObject& so)
{
	auto row_count = so.attribute_read<PVRow>("row_count");

	Squey::PVLinesProperties lp(row_count);
	so.buffer_read("lp_data", lp._colors, row_count);

	return lp;
}
