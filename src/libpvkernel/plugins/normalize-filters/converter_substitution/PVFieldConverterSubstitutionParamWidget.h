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

#ifndef PVFIELDSUBSTITUTIONPARAMWIDGET_H
#define PVFIELDSUBSTITUTIONPARAMWIDGET_H

#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

class QAction;
class QCheckBox;
class QGroupBox;
class QLineEdit;
class QWidget;
class QTableWidget;

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVWidgets
{
class QKeySequenceWidget;
}

namespace PVFilter
{

class PVFieldConverterSubstitutionParamWidget : public PVFieldsConverterParamWidget
{
	Q_OBJECT

  public:
	PVFieldConverterSubstitutionParamWidget();

  public:
	QAction* get_action_menu(QWidget* parent) override;
	QWidget* get_param_widget() override;

  private:
	int get_modes() const;
	void populate_substrings_table(const QStringList& map);
	QString serialize_substrings_map() const;

  private Q_SLOTS:
	void update_params();
	void browse_conversion_file();
	void use_default_value_checkbox_changed(int state);
	void invert_layouts();
	void selection_has_changed();

  private:
	void add_new_row();
	void del_selected_rows();
	void move_rows_up();
	void move_rows_down();

  private:
	QPushButton* _invert_button = nullptr;
	QGroupBox* _whole_field_group_box = nullptr;
	QWidget* _param_widget = nullptr;
	QLineEdit* _file_path_line_edit = nullptr;
	QLineEdit* _default_value_line_edit = nullptr;
	QCheckBox* _use_default_value_checkbox = nullptr;
	PVWidgets::QKeySequenceWidget* _separator_char = nullptr;
	PVWidgets::QKeySequenceWidget* _quote_char = nullptr;

	QGroupBox* _substrings_group_box = nullptr;
	QLineEdit* _replace_line_edit = nullptr;
	QLineEdit* _by_line_edit = nullptr;
	QPushButton* _del_button = nullptr;
	QPushButton* _up_button = nullptr;
	QPushButton* _down_button = nullptr;

	QTableWidget* _substrings_table_widget = nullptr;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterSubstitutionParamWidget)
};
}

#endif // PVFIELDSUBSTITUTIONPARAMWIDGET_H
