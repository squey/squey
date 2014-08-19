/**
 * \file PVFieldValueMapperParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDVALUEMAPPERPARAMWIDGET_H
#define PVFIELDVALUEMAPPERPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QAction;
class QWidget;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldConverterValueMapperParamWidget: public PVFieldsConverterParamWidget
{
	Q_OBJECT

public:
	PVFieldConverterValueMapperParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private slots:
	void update_params();

private:
	QAction* _action_menu;
	QWidget* _param_widget;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterValueMapperParamWidget)
};

}

#endif // PVFIELDVALUEMAPPERPARAMWIDGET_H
