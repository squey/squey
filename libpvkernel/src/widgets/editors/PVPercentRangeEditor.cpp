/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/widgets/editors/PVPercentRangeEditor.h>
#include <pvkernel/widgets/PVAbstractRangePicker.h>

#include <pvkernel/core/PVPercentRangeType.h> // for PVPercentRangeType

#include <QBrush>   // for QLinearGradient
#include <QSpinBox> // for QDoubleSpinBox

class QWidget;

/******************************************************************************
 * PVWidgets::PVPercentRangeEditor::PVPercentRangeEditor
 *****************************************************************************/

PVWidgets::PVPercentRangeEditor::PVPercentRangeEditor(QWidget* parent)
    : PVAbstractRangePicker(0., 100., parent)
{
	_min_spinbox->use_floating_point(true);
	_max_spinbox->use_floating_point(true);

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
