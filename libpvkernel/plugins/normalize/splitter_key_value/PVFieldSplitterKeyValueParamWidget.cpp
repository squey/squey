/**
 * \file PVFieldSplitterKeyValueParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterKeyValueParamWidget.h"
#include "PVFieldSplitterKeyValue.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QAction>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterKeyValue()))
{
	_action_menu = new QAction(QString("add Key Value"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldSplitterKeyValueParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	return _param_widget;
}
