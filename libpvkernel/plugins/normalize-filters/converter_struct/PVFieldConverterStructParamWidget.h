/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSUBSTITUTIONPARAMWIDGET_H
#define PVFIELDSUBSTITUTIONPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QAction;
class QWidget;

namespace PVFilter
{

class PVFieldConverterStructParamWidget : public PVFieldsConverterParamWidget
{
	Q_OBJECT

  public:
	PVFieldConverterStructParamWidget();

  public:
	QAction* get_action_menu();
	QWidget* get_param_widget();

  private:
	QAction* _action_menu;
	QWidget* _param_widget;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterStructParamWidget)
};
}

#endif // PVFIELDSUBSTITUTIONPARAMWIDGET_H
