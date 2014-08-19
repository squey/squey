/**
 * \file PVFieldValueMapperParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldConverterValueMapperParamWidget.h"
#include "PVFieldConverterValueMapper.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QAction>
#include <QRadioButton>
#include <QComboBox>
#include <QGroupBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QMessageBox>

/******************************************************************************
 *
 * PVFilter::PVFieldConverterValueMapperParamWidget::PVFieldSplitterCSVParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldConverterValueMapperParamWidget::PVFieldConverterValueMapperParamWidget() :
	PVFieldsConverterParamWidget(PVFilter::PVFieldsConverter_p(new PVFieldConverterValueMapper()))
{
	_action_menu = new QAction(QString("add Value Mapper"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterValueMapperParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldConverterValueMapperParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldValueMapperParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterValueMapperParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldConverterValueMapperParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldValueMapperParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	return _param_widget;
}


void PVFilter::PVFieldConverterValueMapperParamWidget::update_params()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	get_filter()->set_args(args);
    emit args_changed_Signal();
}
