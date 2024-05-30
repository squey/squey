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

#include "PVFieldSplitterLengthParamWidget.h"

#include "PVFieldSplitterLength.h"

#include <QVBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QPushButton>
#include <QAction>

#include <QDebug>

#include <limits>

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::PVFieldSplitterLengthParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterLengthParamWidget::PVFieldSplitterLengthParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterLength()))
{
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterLengthParamWidget::get_param_widget()
{
	/* initialize the children fields count
	 */
	set_child_count(2);
	Q_EMIT nchilds_changed_Signal();

	/* create the parameter editor widget
	 */
	PVCore::PVArgumentList args = get_filter()->get_args();

	auto* _param_widget = new QWidget;

	auto* layout = new QVBoxLayout;
	_param_widget->setLayout(layout);

	auto* grid = new QGridLayout;
	layout->addLayout(grid);

	auto* label = new QLabel("Number of characters", _param_widget);
	grid->addWidget(label, 0, 0);

	auto* length = new QSpinBox(_param_widget);
	length->setRange(0, std::numeric_limits<int>::max());
	length->setValue(args.at(PVFieldSplitterLength::param_length).toInt());
	connect(length, SIGNAL(valueChanged(int)), this, SLOT(update_length(int)));

	grid->addWidget(length, 0, 1);

	label = new QLabel("Count from", _param_widget);
	grid->addWidget(label, 1, 0);

	bool state = args.at(PVFieldSplitterLength::param_from_left).toBool();
	_side = new QPushButton(_param_widget);
	_side->setCheckable(true);
	_side->setFlat(true);
	_side->setChecked(state);
	set_button_text(state);
	connect(_side, &QAbstractButton::toggled, this, &PVFieldSplitterLengthParamWidget::update_side);

	grid->addWidget(_side, 1, 1);

	auto* help = new QLabel(_param_widget);
	help->setWordWrap(true);
	layout->addWidget(help);

	help->setText("<br><b>Examples:</b>\n"
	              "<p>With <i>Number of characters</i> set to <b>5</b> and <i>Count from</i> "
	              "set to <b>left</b>, the string \"123456789\" will be splitted into the "
	              "strings \"12345\" and \"6789\".</p>"
	              "<p>With <i>Number of characters</i> set to <b>3</b> and <i>Count from</i> "
	              "set to <b>right</b>, the string \"123456789\" will be splitted into the "
	              "strings \"123456\" and \"789\".</p>"
	              "<p>With <i>Number of characters</i> set to <b>15</b> and <i>Count from</i> "
	              "set to <b>left</b>, the string \"123456789\" will be splitted into the "
	              "strings \"123456789\" and \"\".</p>");

	return _param_widget;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterLengthParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Length Splitter"), parent);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::update_length
 *****************************************************************************/

void PVFilter::PVFieldSplitterLengthParamWidget::update_length(int value)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args[PVFieldSplitterLength::param_length] = value;

	get_filter()->set_args(args);

	Q_EMIT args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::update_side
 *****************************************************************************/

void PVFilter::PVFieldSplitterLengthParamWidget::update_side(bool state)
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args[PVFieldSplitterLength::param_from_left] = state;

	set_button_text(state);

	get_filter()->set_args(args);

	Q_EMIT args_changed_Signal();
}

/******************************************************************************
 * PVFilter::PVFieldSplitterLengthParamWidget::set_button_text
 *****************************************************************************/

void PVFilter::PVFieldSplitterLengthParamWidget::set_button_text(bool state)
{
	if (state) {
		_side->setText("Left most");
	} else {
		_side->setText("Right most");
	}
}
