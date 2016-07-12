/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

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

	QWidget* _param_widget = new QWidget;

	QVBoxLayout* layout = new QVBoxLayout;
	_param_widget->setLayout(layout);

	QGridLayout* grid = new QGridLayout;
	layout->addLayout(grid);

	QLabel* label = new QLabel("Number of characters", _param_widget);
	grid->addWidget(label, 0, 0);

	QSpinBox* length = new QSpinBox(_param_widget);
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
	connect(_side, SIGNAL(toggled(bool)), this, SLOT(update_side(bool)));

	grid->addWidget(_side, 1, 1);

	QLabel* help = new QLabel(_param_widget);
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
