/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
