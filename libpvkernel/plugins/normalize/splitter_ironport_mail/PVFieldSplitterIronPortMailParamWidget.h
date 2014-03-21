/**
 * \file PVFieldSplitterIronPortMailParamWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#ifndef PVFIELDSPLITTERIRONPORMAILPARAMWIDGET_H
#define PVFIELDSPLITTERIRONPORMAILPARAMWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QAction;
class QWidget;

namespace PVFilter {


class PVFieldSplitterIronPortMailParamWidget: public PVFieldsSplitterParamWidget
{
public:
	PVFieldSplitterIronPortMailParamWidget();

	QAction* get_action_menu() override;

	QWidget* get_param_widget() override;

public:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterIronPortMailParamWidget)

private:
	QAction *_menu_action;
};

}

#endif // PVFIELDSPLITTERIRONPORMAILPARAMWIDGET_H
