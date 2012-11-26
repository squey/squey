/**
 * \file PVFieldSplitterDuplicateParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include "PVFieldSplitterDuplicateParamWidget.h"
#include "PVFieldDuplicate.h"
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QLabel>
#include <QHBoxLayout>
#include <QSpacerItem>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterDuplicateParamWidget::PVFieldSplitterDuplicateParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldDuplicate()))
{
	_action_menu = new QAction(QString("add Duplicate Splitter"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterDuplicateParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldSplitterDuplicateParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterDuplicateParamWidget::get_param_widget()     start\n");

	//get args
	PVCore::PVArgumentList l =  get_filter()->get_args();

	uint32_t n = l["n"].toUInt();
	updateNChilds(n);

	_param_widget = new QWidget();

	QHBoxLayout* layout = new QHBoxLayout(_param_widget);

	_param_widget->setLayout(layout);
	_param_widget->setObjectName("splitter");

	// title
	QLabel* duplications_label = new QLabel(tr("Number of duplications:"));
	layout->addWidget(duplications_label);

	// Spin box
	_duplications_spin_box = new QSpinBox();
	_duplications_spin_box->setMinimum(2);
	_duplications_spin_box->setValue(n);
	layout->addWidget(_duplications_spin_box);

	connect(_duplications_spin_box, SIGNAL(valueChanged(int)), this, SLOT(updateNChilds(int)));

	return _param_widget;
}

void PVFilter::PVFieldSplitterDuplicateParamWidget::updateNChilds(int n)
{
	PVCore::PVArgumentList args = get_filter()->get_args();
	args["n"] = n;
	get_filter()->set_args(args);
	set_child_count(n);
    emit args_changed_Signal();
	emit nchilds_changed_Signal();
}
