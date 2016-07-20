/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSTRUCTPARAMWIDGET_H
#define PVFIELDSTRUCTPARAMWIDGET_H

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
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private:
	QWidget* _param_widget;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterStructParamWidget)
};
}

#endif // PVFIELDSTRUCTPARAMWIDGET_H
