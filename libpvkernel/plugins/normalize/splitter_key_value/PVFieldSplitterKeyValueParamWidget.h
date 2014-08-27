/**
 * \file PVFieldSplitterKeyValueParamWidget.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
#define PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

class QAction;
class QWidget;

namespace PVFilter {

class PVFieldSplitterKeyValueParamWidget: public PVFieldsSplitterParamWidget
{
	Q_OBJECT

public:
	PVFieldSplitterKeyValueParamWidget();

public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

private:
	QAction* _action_menu;
	QWidget* _param_widget;

private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterKeyValueParamWidget)
};

}

#endif // PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
