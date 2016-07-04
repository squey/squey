/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H
#define PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QWidget;
class QAction;
class QRadioButton;
class QComboBox;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter
{

class PVFieldConverterGUIDToIPParamWidget : public PVFieldsConverterParamWidget
{
	Q_OBJECT

  public:
	PVFieldConverterGUIDToIPParamWidget();

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private slots:
	void update_params();

  private:
	QWidget* _param_widget;

	QRadioButton* _ipv4;
	QRadioButton* _ipv6;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterGUIDToIPParamWidget)
};
}

#endif // PVFIELDCONVERTERGUIDTOIPPARAMWIDGET_H
