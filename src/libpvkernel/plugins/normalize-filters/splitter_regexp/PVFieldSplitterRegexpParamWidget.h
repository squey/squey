/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
	QWidget* param_widget = nullptr;
	int id = 0;

	// widget showed
	QLineEdit* expression_lineEdit = nullptr;
	QLabel* child_count_text = nullptr;
	QTextEdit* validator_textEdit = nullptr;
	QTableWidget* table_validator_TableWidget = nullptr;
	QPushButton* btn_apply = nullptr;
	QCheckBox* fullline_checkBox = nullptr;

	bool expressionChanged = false;

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
