/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H
#define PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QWidget>

namespace PVFilter
{

class PVFieldSplitterMacAddressParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterMacAddressParamWidget();

  public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

	size_t force_number_children() { return 2; }

  private:
	QAction* _action_menu;
	QWidget* _param_widget;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterMacAddressParamWidget)
};
}

#endif // PVFIELDSPLITTERMACADDRESSPARAMWIDGET_H