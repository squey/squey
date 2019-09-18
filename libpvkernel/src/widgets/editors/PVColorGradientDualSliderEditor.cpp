/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
