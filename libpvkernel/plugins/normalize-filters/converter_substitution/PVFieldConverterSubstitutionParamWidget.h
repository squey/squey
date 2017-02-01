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
	QPushButton* _invert_button;
	QGroupBox* _whole_field_group_box;
	QWidget* _param_widget;
	QLineEdit* _file_path_line_edit;
	QLineEdit* _default_value_line_edit;
	QCheckBox* _use_default_value_checkbox;
	PVWidgets::QKeySequenceWidget* _separator_char;
	PVWidgets::QKeySequenceWidget* _quote_char;

	QGroupBox* _substrings_group_box;
	QLineEdit* _replace_line_edit;
	QLineEdit* _by_line_edit;
	QPushButton* _del_button;
	QPushButton* _up_button;
	QPushButton* _down_button;

	QTableWidget* _substrings_table_widget;

  private:
	CLASS_REGISTRABLE_NOCOPY(PVFieldConverterSubstitutionParamWidget)
};
}

#endif // PVFIELDSUBSTITUTIONPARAMWIDGET_H
