/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFIELDSPLITTERREGEXPPARAMWIDGET_H
#define PVFIELDSPLITTERREGEXPPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

#include <QBoxLayout>
#include <QCheckBox>
#include <QLabel>
#include <QObject>
#include <QAction>
#include <QLineEdit>
#include <QTextEdit>
#include <QTableWidget>
#include <QPushButton>
#include <QStringList>
#include <QRegExp>

namespace PVFilter
{

class PVFieldSplitterRegexpParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT;

  private:
	QWidget* param_widget;
	int id;

	// widget showed
	QLineEdit* expression_lineEdit;
	QLabel* child_count_text;
	QTextEdit* validator_textEdit;
	QTableWidget* table_validator_TableWidget;
	QPushButton* btn_apply;
	QCheckBox* fullline_checkBox;

	bool expressionChanged;

	void initWidget();

  public:
	PVFieldSplitterRegexpParamWidget();
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

	void set_id(int id_param) override { id = id_param; }

	void update_data_display() override;

	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterRegexpParamWidget)
  public Q_SLOTS:
	void slotUpdateTableValidator();
	void slotExpressionChanged();
	void slotFullineChanged(int state);

  Q_SIGNALS:
	void data_changed();
	void signalRefreshView();
};
}

#endif
