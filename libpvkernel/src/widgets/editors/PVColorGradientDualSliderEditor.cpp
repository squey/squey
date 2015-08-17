/**
 * \file PVColorGradientDualSliderEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <cmath>

#include <QtCore>
#include <QtWidgets>


#include <pvkernel/core/general.h>
#include <pvkernel/widgets/editors/PVColorGradientDualSliderEditor.h>

/******************************************************************************
 *
 * PVWidgets::PVColorGradientDualSliderEditor::PVColorGradientDualSliderEditor
 *
 *****************************************************************************/
PVWidgets::PVColorGradientDualSliderEditor::PVColorGradientDualSliderEditor(QWidget *parent) :
	PVColorPicker(parent)
{
	set_selection_mode(PVColorPicker::SelectionInterval);
	set_allow_empty_interval(false);
	set_x0(HSV_COLOR_GREEN);
	set_x1(HSV_COLOR_RED);

	// AG: that's the one million dollars line that allows the widget to keep the focus when it is clicked
	// and to not give the focus to the underlying item (in the table view) ; because in such a case, it is
	// then hidden !!!!!!
	// (historical now... kept for the time it took to figure this out)
	//setFocusPolicy(Qt::StrongFocus);
}

PVCore::PVColorGradientDualSliderType PVWidgets::PVColorGradientDualSliderEditor::get_values() const
{
	PVCore::PVColorGradientDualSliderType ret;
	double pos[2];
	pos[0] = (double)(interval_left().h() -x0())/(x_interval());
	pos[1] = (double)(interval_right().h()-x0())/(x_interval());
	ret.set_positions(pos);
	return ret;
}

void PVWidgets::PVColorGradientDualSliderEditor::set_values(PVCore::PVColorGradientDualSliderType v)
{
	const double* pos = v.get_positions();
	set_interval((uint8_t)(pos[0]*x_interval()+x0()), (uint8_t)(pos[1]*x_interval()+x0()));
}
