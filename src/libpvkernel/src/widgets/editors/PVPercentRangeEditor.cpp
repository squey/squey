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
	lg.setColorAt(0.0, Qt::blue);
	lg.setColorAt(0.25, Qt::cyan);
	lg.setColorAt(0.5, Qt::green);
	lg.setColorAt(0.75, Qt::yellow);
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
