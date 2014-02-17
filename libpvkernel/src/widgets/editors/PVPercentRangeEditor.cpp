/**
 * \file PVPercentRangeEditor.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <cmath>

#include <QtCore>
#include <QtGui>

#include <pvkernel/core/general.h>
#include <pvkernel/widgets/editors/PVPercentRangeEditor.h>

/******************************************************************************
 * PVWidgets::PVPercentRangeEditor::PVPercentRangeEditor
 *****************************************************************************/

PVWidgets::PVPercentRangeEditor::PVPercentRangeEditor(QWidget *parent) :
	PVAbstractRangePicker(0., 100., parent)
{
	get_min_spinbox()->setDecimals(1);
	get_max_spinbox()->setDecimals(1);

	get_min_spinbox()->setSingleStep(0.1);
	get_max_spinbox()->setSingleStep(0.1);

	get_min_spinbox()->setSuffix(" %");
	get_max_spinbox()->setSuffix(" %");

	QLinearGradient lg;
	lg.setColorAt(0.0, Qt::green);
	lg.setColorAt(0.5, Qt::yellow);
	lg.setColorAt(1.0, Qt::red);

	set_gradient(lg);
}

/******************************************************************************
 * PVWidgets::PVPercentRangeEditor::get_values
 *****************************************************************************/

PVCore::PVPercentRangeType PVWidgets::PVPercentRangeEditor::get_values() const
{
	PVCore::PVPercentRangeType ret;

	double pos[2];
	pos[0] = get_range_min() * 0.01;
	pos[1] = get_range_max() * 0.01;
	ret.set_values(pos);

	return ret;
}

/******************************************************************************
 * PVWidgets::PVPercentRangeEditor::get_values
 *****************************************************************************/

void PVWidgets::PVPercentRangeEditor::set_values(const PVCore::PVPercentRangeType& r)
{
	const double* pos = r.get_values();

	set_range_min(pos[0] * 100.);
	set_range_max(pos[1] * 100.);
}
