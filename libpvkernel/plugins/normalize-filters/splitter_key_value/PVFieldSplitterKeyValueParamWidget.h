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

#ifndef PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
#define PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QAction;
class QWidget;
class QListWidget;
class QLineEdit;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVFilter
{

class PVFieldSplitterKeyValueParamWidget : public PVFieldsSplitterParamWidget
{
	Q_OBJECT

  public:
	PVFieldSplitterKeyValueParamWidget();

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private Q_SLOTS:
	void update_params();
	void add_new_key();
	void del_keys();
	void move_key_down();
	void move_key_up();
	void update_children_count();
	void copy_keys();
	void paste_keys();
	void selection_has_changed();

  private:
	void add_new_keys(QStringList& keys);

  private:
	QWidget* _param_widget = nullptr;

	PVWidgets::QKeySequenceWidget* _quote_char = nullptr;
	QLineEdit* _separator_char_lineedit = nullptr;
	QLineEdit* _affectation_operator_lineedit = nullptr;
	QPushButton* _del_button = nullptr;
	QPushButton* _up_button = nullptr;
	QPushButton* _down_button = nullptr;
	QPushButton* _copy_button = nullptr;

	QListWidget* _keys_list = nullptr;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterKeyValueParamWidget)
};
}

#endif // PVFIELDSPLITTERKEYVALUEPARAMWIDGET_H
