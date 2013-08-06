/**
 * \file PVFieldConverterGUIDToIPParamWidget.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H
#define PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QWidget;
class QAction;
class QRadioButton;
class QComboBox;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter {

class PVFieldConverterGUIDToIPParamWidget: public PVFieldsConverterParamWidget
{
	Q_OBJECT

public:
	PVFieldConverterGUIDToIPParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private slots:
	void update_params();

private:
	QAction* _action_menu;
	QWidget* _param_widget;

	QRadioButton* _ipv4;
	QRadioButton* _ipv6;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterGUIDToIPParamWidget)
};

}

#endif // PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H
