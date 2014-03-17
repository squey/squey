/**
 * \file PVFieldSplitterMacAddressParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterMacAddressParamWidget.h"
#include "PVFieldMacAddress.h"

#include <QAction>
#include <QLabel>
#include <QHBoxLayout>

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::PVFieldSplitterCSVParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterMacAddressParamWidget::PVFieldSplitterMacAddressParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldMacAddress()))
{
	_action_menu = new QAction(QString("add Mac Address Splitter"), NULL);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterMacAddressParamWidget::get_action_menu()
{
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterMacAddressParamWidget::get_param_widget()
{
	//get args
	PVCore::PVArgumentList al =  get_filter()->get_args();

	bool uppercased = al[PVFieldMacAddress::uppercased_str].toBool();

	_param_widget = new QWidget();

	QHBoxLayout* layout = new QHBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	QLabel* label = new QLabel(tr("Case:"));
	layout->addWidget(label);

	_case_button = new QPushButton();
	_case_button->setCheckable(true);
	layout->addWidget(_case_button);

	update_case(uppercased);

	_case_button->setChecked(uppercased);

	connect(_case_button, SIGNAL(toggled(bool)), this, SLOT(update_case(bool)));

	return _param_widget;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterMacAddressParamWidget::update_case
 *****************************************************************************/

void PVFilter::PVFieldSplitterMacAddressParamWidget::update_case(bool uppercased)
{
	if (uppercased) {
		_case_button->setText("Uppercase");
	} else {
		_case_button->setText("Lowercase");
	}

	PVCore::PVArgumentList args = get_filter()->get_args();
	args[PVFieldMacAddress::uppercased_str] = uppercased;
	get_filter()->set_args(args);
	emit args_changed_Signal();
}
