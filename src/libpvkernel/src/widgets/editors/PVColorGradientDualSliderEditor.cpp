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

#include <pvkernel/widgets/PVColorPicker.h> // for PVColorPicker, etc
#include <pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h>

#include <pvkernel/core/PVColorGradientDualSliderType.h>
#include <pvkernel/core/PVHSVColor.h> // for PVHSVColor, HSV_COLOR_GREEN, etc

#include <cstdint> // for uint8_t

class QWidget;

/******************************************************************************
 *
 * PVWidgets::PVColorGradientDualSliderEditor::PVColorGradientDualSliderEditor
 *
 *****************************************************************************/
PVWidgets::PVColorGradientDualSliderEditor::PVColorGradientDualSliderEditor(QWidget* parent)
    : PVColorPicker(parent)
{
	set_selection_mode(PVColorPicker::SelectionInterval);
	set_allow_empty_interval(false);
	set_x0(HSV_COLOR_BLUE.h());
	set_x1(HSV_COLOR_RED.h());
}

PVCore::PVColorGradientDualSliderType PVWidgets::PVColorGradientDualSliderEditor::get_values() const
{
	PVCore::PVColorGradientDualSliderType ret;
	double pos[2];
	pos[0] = (double)(interval_left().h() - x0()) / (x_interval());
	pos[1] = (double)(interval_right().h() - x0()) / (x_interval());
	ret.set_positions(pos);
	return ret;
}

void PVWidgets::PVColorGradientDualSliderEditor::set_values(PVCore::PVColorGradientDualSliderType v)
{
	const double* pos = v.get_positions();
	set_interval((PVCore::PVHSVColor)(pos[0] * x_interval() + x0()),
	             (PVCore::PVHSVColor)(pos[1] * x_interval() + x0()));
}
