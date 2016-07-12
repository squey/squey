/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#ifndef PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H
#define PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QPushButton;

namespace PVFilter
{

class PVFieldSplitterLengthParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterLengthParamWidget();

  public:
	QWidget* get_param_widget() override;
	QAction* get_action_menu(QWidget* parent) override;

  public Q_SLOTS:
	void update_length(int value);
	void update_side(bool state);

  protected:
	void set_button_text(bool state);

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterLengthParamWidget)

  private:
	QPushButton* _side;
};
}

#endif // PVFILTER_PVFIELDSPLITTERLENGTHPARAMWIDGET_H
