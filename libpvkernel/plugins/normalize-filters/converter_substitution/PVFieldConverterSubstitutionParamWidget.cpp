//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVFieldConverterSubstitutionParamWidget.h"
#include "PVFieldConverterSubstitution.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <QAction>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QGroupBox>
#include <QTableWidget>
#include <QHeaderView>

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::PVFieldConverterSubstitutionParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldConverterSubstitutionParamWidget::PVFieldConverterSubstitutionParamWidget()
    : PVFieldsConverterParamWidget(
          PVFilter::PVFieldsConverter_p(new PVFieldConverterSubstitution()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldConverterSubstitutionParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Substitution Converter"), parent);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldConverterSubstitutionParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSubstitutionParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	size_t modes = args["modes"].toUInt();

	/**
	 * whole field mode
	 */
	_whole_field_group_box = new QGroupBox("Rewrite entier fields");
	_whole_field_group_box->setCheckable(true);
	_whole_field_group_box->setChecked(modes & PVFieldConverterSubstitution::WHOLE_FIELD);

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);
	QVBoxLayout* entier_field_layout = new QVBoxLayout();
	_whole_field_group_box->setLayout(entier_field_layout);

	QHBoxLayout* file_layout = new QHBoxLayout();

	QLabel* file_label = new QLabel("Conversion file:");
	_file_path_line_edit = new QLineEdit();
	_file_path_line_edit->setReadOnly(true);
	_file_path_line_edit->setText(args["path"].toString());

	QPushButton* browse_pushbutton = new QPushButton("...");

	file_layout->addWidget(file_label);
	file_layout->addWidget(_file_path_line_edit);
	file_layout->addWidget(browse_pushbutton);

	// Fields separator
	QHBoxLayout* fields_separator_layout = new QHBoxLayout();
	QLabel* separator_label = new QLabel(tr("Fields separator:"));
	_separator_char = new PVWidgets::QKeySequenceWidget();
	_separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_separator_char->setKeySequence(QKeySequence(args["sep"].toString()));
	_separator_char->setMaxNumKey(1);
	fields_separator_layout->addWidget(separator_label);
	fields_separator_layout->addWidget(_separator_char);
	fields_separator_layout->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// Quote character
	QHBoxLayout* quote_character_layout = new QHBoxLayout();
	QLabel* quote_label = new QLabel(tr("Quote character:"));
	_quote_char = new PVWidgets::QKeySequenceWidget();
	_quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_quote_char->setKeySequence(QKeySequence(args["quote"].toString()));
	_quote_char->setMaxNumKey(1);
	quote_character_layout->addWidget(quote_label);
	quote_character_layout->addWidget(_quote_char);
	quote_character_layout->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	QHBoxLayout* default_value_layout = new QHBoxLayout();

	_use_default_value_checkbox = new QCheckBox();
	_use_default_value_checkbox->setChecked(args["use_default_value"].toBool());
	QLabel* default_value_label = new QLabel("Default value:");
	_default_value_line_edit = new QLineEdit();
	_default_value_line_edit->setText(args["default_value"].toString());
	_default_value_line_edit->setEnabled(false);

	default_value_layout->addWidget(_use_default_value_checkbox);
	default_value_layout->addWidget(default_value_label);
	default_value_layout->addWidget(_default_value_line_edit);

	entier_field_layout->addLayout(file_layout);
	entier_field_layout->addLayout(fields_separator_layout);
	entier_field_layout->addLayout(quote_character_layout);
	entier_field_layout->addLayout(default_value_layout);
	entier_field_layout->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	connect(_whole_field_group_box, &QGroupBox::toggled, this,
	        &PVFieldConverterSubstitutionParamWidget::update_params);
	connect(browse_pushbutton, &QPushButton::clicked, this,
	        &PVFieldConverterSubstitutionParamWidget::browse_conversion_file);
	connect(_default_value_line_edit, &QLineEdit::textChanged, this,
	        &PVFieldConverterSubstitutionParamWidget::update_params);
	connect(_use_default_value_checkbox, &QCheckBox::stateChanged, this,
	        &PVFieldConverterSubstitutionParamWidget::use_default_value_checkbox_changed);
	connect(_separator_char, &PVWidgets::QKeySequenceWidget::keySequenceChanged, this,
	        &PVFieldConverterSubstitutionParamWidget::update_params);
	connect(_quote_char, &PVWidgets::QKeySequenceWidget::keySequenceChanged, this,
	        &PVFieldConverterSubstitutionParamWidget::update_params);

	/**
	 * Substrings mode
	 */
	_substrings_group_box = new QGroupBox("Rewrite substrings");
	_substrings_group_box->setCheckable(true);
	_substrings_group_box->setChecked(modes & PVFieldConverterSubstitution::SUBSTRINGS);

	QVBoxLayout* substrings_layout = new QVBoxLayout();

	_replace_line_edit = new QLineEdit;
	_by_line_edit = new QLineEdit;

	QHBoxLayout* from_to_layout = new QHBoxLayout();
	from_to_layout->addWidget(_replace_line_edit);
	from_to_layout->addWidget(new QLabel(" -> "));
	from_to_layout->addWidget(_by_line_edit);
	QPushButton* add_button = new QPushButton("Add");
	add_button->setEnabled(false);
	from_to_layout->addWidget(add_button);

	substrings_layout->addLayout(from_to_layout);

	QHBoxLayout* substrings_table_widget_layout = new QHBoxLayout();

	_substrings_table_widget = new QTableWidget;
	_substrings_table_widget->setColumnCount(2);
	_substrings_table_widget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	_substrings_table_widget->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_substrings_table_widget->setSelectionBehavior(QAbstractItemView::SelectRows);
	_substrings_table_widget->setEditTriggers(QAbstractItemView::NoEditTriggers);
	_substrings_table_widget->verticalHeader()->setVisible(false);
	_substrings_table_widget->setHorizontalHeaderLabels({"Replace", "By"});
	populate_substrings_table(
	    PVCore::deserialize_base64<QStringList>(args["substrings_map"].toString()));

	QVBoxLayout* buttons_layout = new QVBoxLayout();

	_del_button = new QPushButton(tr("Delete"));
	_up_button = new QPushButton(tr("Move up"));
	_down_button = new QPushButton(tr("Move down"));
	QPushButton* import_button = new QPushButton(tr("Import"));
	import_button->setEnabled(false);
	QPushButton* export_button = new QPushButton(tr("Export"));
	export_button->setEnabled(false);

	_del_button->setIcon(QIcon(":/red-cross"));
	_up_button->setIcon(QIcon(":/go-up"));
	_down_button->setIcon(QIcon(":/go-down"));
	import_button->setIcon(QIcon(":/import_file"));
	export_button->setIcon(QIcon(":/export_file"));

	buttons_layout->addWidget(_del_button);
	buttons_layout->addWidget(_up_button);
	buttons_layout->addWidget(_down_button);
	buttons_layout->addWidget(import_button);
	buttons_layout->addWidget(export_button);

	substrings_table_widget_layout->addWidget(_substrings_table_widget);
	substrings_table_widget_layout->addLayout(buttons_layout);

	substrings_layout->addLayout(substrings_table_widget_layout);

	_substrings_group_box->setLayout(substrings_layout);

	QHBoxLayout* invert_layout = new QHBoxLayout();
	_invert_button = new QPushButton("Invert order");
	_invert_button->setCheckable(true);
	invert_layout->addWidget(_invert_button);
	invert_layout->addStretch();

	layout->addLayout(invert_layout);
	layout->addWidget(_whole_field_group_box);
	layout->addWidget(_substrings_group_box);

	connect(_substrings_group_box, &QGroupBox::toggled, this,
	        &PVFieldConverterSubstitutionParamWidget::update_params);
	connect(add_button, &QPushButton::clicked, this,
	        &PVFilter::PVFieldConverterSubstitutionParamWidget::add_new_row);
	connect(_del_button, &QPushButton::clicked, this,
	        &PVFilter::PVFieldConverterSubstitutionParamWidget::del_selected_rows);
	connect(_up_button, &QPushButton::clicked, this,
	        &PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_up);
	connect(_down_button, &QPushButton::clicked, this,
	        &PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_down);
	connect(_replace_line_edit, &QLineEdit::textChanged,
	        [=,this]() { add_button->setEnabled(not _replace_line_edit->text().isEmpty()); });
	connect(_invert_button, &QPushButton::toggled, this,
	        &PVFieldConverterSubstitutionParamWidget::invert_layouts);
	connect(_substrings_table_widget, &QTableWidget::itemSelectionChanged, this,
	        &PVFieldConverterSubstitutionParamWidget::selection_has_changed);

	_invert_button->setChecked(args["invert_order"].toBool());

	return _param_widget;
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::invert_layouts()
{
	QVBoxLayout* layout = (QVBoxLayout*)_param_widget->layout();

	QWidget* widget = layout->itemAt(1)->widget();
	layout->removeWidget(widget);
	layout->addWidget(widget);

	update_params();
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::selection_has_changed()
{
	QModelIndexList selection = _substrings_table_widget->selectionModel()->selectedRows();

	if (selection.isEmpty()) {
		_del_button->setDisabled(true);
		_up_button->setDisabled(true);
		_down_button->setDisabled(true);

		return;
	}

	QAbstractItemModel* model = _substrings_table_widget->model();

	/* must sort keys to make first() have the smallest row index and last() have the greatest
	 * row index
	 */
	std::sort(selection.begin(), selection.end(), std::less<QModelIndex>());

	_up_button->setEnabled(selection.first().row() > 0);
	_down_button->setEnabled(selection.last().row() < (model->rowCount() - 1));

	_del_button->setEnabled(true);
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::populate_substrings_table(
    const QStringList& map)
{
	for (int i = 0; i < ((map.size() / 2) * 2) / 2; i++) {
		_substrings_table_widget->insertRow(i);
		_substrings_table_widget->setItem(i, 0, new QTableWidgetItem(map[i * 2]));
		_substrings_table_widget->setItem(i, 1, new QTableWidgetItem(map[i * 2 + 1]));
	}
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::add_new_row
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::add_new_row()
{
	int row_count = _substrings_table_widget->rowCount();

	_substrings_table_widget->insertRow(row_count);
	_substrings_table_widget->setItem(row_count, 0,
	                                  new QTableWidgetItem(_replace_line_edit->text()));
	_substrings_table_widget->setItem(row_count, 1, new QTableWidgetItem(_by_line_edit->text()));

	_replace_line_edit->clear();
	_by_line_edit->clear();

	update_params();
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::del_selected_rows
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::del_selected_rows()
{
	QItemSelectionModel* select_model = _substrings_table_widget->selectionModel();
	QModelIndexList selected_rows = select_model->selectedRows();
	int selected_row_count = selected_rows.count();

	for (int i = selected_row_count; i > 0; i--) {
		_substrings_table_widget->removeRow(selected_rows.at(i - 1).row());
	}

	update_params();
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_up
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_up()
{
	QItemSelectionModel* select_model = _substrings_table_widget->selectionModel();
	QModelIndexList selected_rows = select_model->selectedRows();
	std::sort(selected_rows.begin(), selected_rows.end(),
	      [](const QModelIndex& i1, const QModelIndex& i2) { return i1.row() < i2.row(); });
	int selected_row_count = selected_rows.count();

	/* use temporarily MultiSelection to keep origin selection in spite of item move
	 */
	_substrings_table_widget->setSelectionMode(QAbstractItemView::MultiSelection);

	for (int i = 0; i < selected_row_count; i++) {
		int row_index = selected_rows.at(i).row();

		if (row_index > 0) {

			QTableWidgetItem* item1 = _substrings_table_widget->takeItem(row_index, 0);
			QTableWidgetItem* item2 = _substrings_table_widget->takeItem(row_index, 1);

			_substrings_table_widget->insertRow(row_index - 1);
			_substrings_table_widget->setItem(row_index - 1, 0, item1);
			_substrings_table_widget->setItem(row_index - 1, 1, item2);
			_substrings_table_widget->selectRow(row_index - 1);

			_substrings_table_widget->removeRow(row_index + 1);
		}
	}

	/* restore original selection mode
	 */
	_substrings_table_widget->setSelectionMode(QAbstractItemView::ExtendedSelection);

	update_params();
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_down
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::move_rows_down()
{
	QItemSelectionModel* select_model = _substrings_table_widget->selectionModel();
	QModelIndexList selected_rows = select_model->selectedRows();
	std::sort(selected_rows.begin(), selected_rows.end(),
	      [](const QModelIndex& i1, const QModelIndex& i2) { return i1.row() < i2.row(); });
	int selected_row_count = selected_rows.count();

	/* use temporarily MultiSelection to keep origin selection in spite of item move
	 */
	_substrings_table_widget->setSelectionMode(QAbstractItemView::MultiSelection);

	for (int i = selected_row_count; i > 0; i--) {
		int row_index = selected_rows.at(i - 1).row();

		if (row_index < _substrings_table_widget->rowCount() - 1) {

			QTableWidgetItem* item1 = _substrings_table_widget->takeItem(row_index, 0);
			QTableWidgetItem* item2 = _substrings_table_widget->takeItem(row_index, 1);

			_substrings_table_widget->insertRow(row_index + 2);
			_substrings_table_widget->setItem(row_index + 2, 0, item1);
			_substrings_table_widget->setItem(row_index + 2, 1, item2);
			_substrings_table_widget->selectRow(row_index + 2);

			_substrings_table_widget->removeRow(row_index);
		}
	}

	/* restore original selection mode
	 */
	_substrings_table_widget->setSelectionMode(QAbstractItemView::ExtendedSelection);

	update_params();
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_substrings_map
 *
 *****************************************************************************/
QString PVFilter::PVFieldConverterSubstitutionParamWidget::serialize_substrings_map() const
{
	QStringList map;

	for (int i = 0; i < _substrings_table_widget->rowCount(); i++) {
		map << _substrings_table_widget->item(i, 0)->text();
		map << _substrings_table_widget->item(i, 1)->text();
	}

	return PVCore::serialize_base64(map);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_modes
 *
 *****************************************************************************/
int PVFilter::PVFieldConverterSubstitutionParamWidget::get_modes() const
{
	return _whole_field_group_box->isChecked() + (_substrings_group_box->isChecked() * 2);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::update_params
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::update_params()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args["modes"] = get_modes();
	args["invert_order"] = _invert_button->isChecked();

	// whole fields mode
	args["path"] = _file_path_line_edit->text();
	args["default_value"] = _default_value_line_edit->text();
	args["use_default_value"] = _use_default_value_checkbox->isChecked();
	args["sep"] = _separator_char->keySequence().toString();
	args["quote"] = _quote_char->keySequence().toString();

	// substrings mode
	args["substrings_map"] = serialize_substrings_map();

	get_filter()->set_args(args);
	Q_EMIT args_changed_Signal();
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::browse_conversion_file()
{
	PVWidgets::PVFileDialog fd;

	QString filename = fd.getOpenFileName(nullptr, tr("Open File"), "", tr("Files (*.*)"));
	if (!filename.isEmpty()) {
		_file_path_line_edit->setText(filename);
	}

	update_params();
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::use_default_value_checkbox_changed(
    int state)
{
	_default_value_line_edit->setEnabled(state == Qt::Checked);

	update_params();
}
