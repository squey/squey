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
	QWidget* param_widget;
	QSpinBox* _child_number_edit; //!< Widget to select number of child (number of column in csv)
	PVWidgets::QKeySequenceWidget* separator_text;
	PVWidgets::QKeySequenceWidget* quote_text;
	// QLineEdit* separator_text;
	QLabel* _recommands_label;
	int id;

  public:
	QWidget* get_param_widget() override;
	QAction* get_action_menu(QWidget* parent) override;

	void set_id(int id_param) override { id = id_param; }

  private:
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
