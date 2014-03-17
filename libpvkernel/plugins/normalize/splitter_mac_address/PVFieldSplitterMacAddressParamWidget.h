/**
 * \file PVFieldSplitterMacAddressParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H
#define PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QWidget>
#include <QPushButton>

namespace PVFilter {

class PVFieldSplitterMacAddressParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterMacAddressParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

	size_t force_number_children() {
		return 2;
	}

private slots:
	void update_case(bool uppecased);

private:
	QAction*     _action_menu;
	QWidget*     _param_widget;
	QPushButton* _case_button;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterMacAddressParamWidget)
};

}

#endif // PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H
