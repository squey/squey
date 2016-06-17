/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QBoxLayout>
#include <QLabel>
#include <QObject>
#include <QAction>

#include <pvkernel/widgets/qkeysequencewidget.h>

class QSpinBox;

namespace PVFilter
{

class PVFieldSplitterCSVParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterCSVParamWidget();
	// PVFieldSplitterCSVParamWidget(const PVFieldSplitterCSVParamWidget& src);

  private:
	QAction* action_menu;
	QWidget* param_widget;
	QSpinBox* _child_number_edit; //!< Widget to select number of child (number of column in csv)
	PVWidgets::QKeySequenceWidget* separator_text;
	PVWidgets::QKeySequenceWidget* quote_text;
	// QLineEdit* separator_text;
	QLabel* _recommands_label;
	int id;

  public:
	QWidget* get_param_widget();
	QAction* get_action_menu();

	void set_id(int id_param) { id = id_param; }
	void update_data_display();

  private:
	void init();
	void update_recommanded_nfields();

	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterCSVParamWidget)

  public Q_SLOTS:
	// void updateSeparator(const QString &sep);
	void updateSeparator(QKeySequence key);
	void updateQuote(QKeySequence key);
	void updateNChilds();

  Q_SIGNALS:
	void signalRefreshView();
};
}

#endif
