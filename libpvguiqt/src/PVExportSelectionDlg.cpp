/**
 * \file PVExportSelectionDlg.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpacerItem>
#include <QMessageBox>

#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <picviz/PVSelection.h>
#include <picviz/PVView.h>

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVProgressBox.h>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(Picviz::PVAxesCombination& custom_axes_combination,
													Picviz::PVView& view, QWidget* parent /* = 0 */) :
	QFileDialog(parent)
{
	setWindowTitle(tr("Export selection"));
	setAcceptMode(QFileDialog::AcceptSave);

	// Get the layout from the QFileDialog
	// Layout is:
	// -----------------main_layout--------------------------------
	// |           name bar where you can put filename/filedir    |
	// ------------------------------------------------------------
	// | dir tree | list of files                                 |
	// ------------------------------------------------------------
	// |filename_label | filename | valide_button                 |
	// ------------------------------------------------------------
	// |filetype_label | filetype | cancel_button                 |
	// ------------------------------------------------------------
	// |----------------group_box---------------------------------|
	// ||                  export_layout                         ||
	// |----------------------------------------------------------|
	// ------------------------------------------------------------
	QGridLayout* main_layout = static_cast<QGridLayout *>(layout());

	QGroupBox* group_box = new QGroupBox();
	main_layout->addWidget(group_box, 4, 0, 1, 3);

	// Layout for export_layout is:
	// --------------------export_layout---------------------------
	// |----------------------------------------------------------|
	// ||  left_layout              |   right layout             ||
	// |----------------------------------------------------------|
	// ------------------------------------------------------------
	QHBoxLayout* export_layout = new QHBoxLayout();
	group_box->setLayout(export_layout);
	QVBoxLayout* left_layout = new QVBoxLayout();
	QVBoxLayout* right_layout = new QVBoxLayout();
	export_layout->addLayout(left_layout);
	export_layout->addLayout(right_layout);

	/// left_layout

	// Export column name
	_columns_header = new QCheckBox("Export column names as header");
	_columns_header->setCheckState(Qt::CheckState::Checked);
	left_layout->addWidget(_columns_header);

	// Define csv specific character
	QFormLayout* char_layout = new QFormLayout();
	char_layout->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);

	// Separator character
	_separator_char = new PVWidgets::QKeySequenceWidget();
	_separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_separator_char->setKeySequence(QKeySequence(","));
	_separator_char->setMaxNumKey(1);
	char_layout->addRow(tr("Fields separator:"), _separator_char);

	// Quote character
	_quote_char = new PVWidgets::QKeySequenceWidget();
	_quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_quote_char->setKeySequence(QKeySequence("\""));
	_quote_char->setMaxNumKey(1);
	char_layout->addRow(tr("Quote character:"), _quote_char);

	left_layout->addLayout(char_layout);

	/// right_layout

	// Use all axes combination
	_all_axis = new QRadioButton("Use all axes combination");
	right_layout->addWidget(_all_axis);

	// Use current axes combination
	_current_axis = new QRadioButton("Use current axes combination");
	_current_axis->setChecked(true);
	right_layout->addWidget(_current_axis);

	// Use custom axes combination
	QHBoxLayout* custom_axes_combination_layout = new QHBoxLayout();

	_custom_axis = new QRadioButton("Use custom axes combination");
	_edit_axes_combination = new QPushButton("Edit");
	_edit_axes_combination->setEnabled(_custom_axis->isChecked());

	custom_axes_combination_layout->addWidget(_custom_axis);
	custom_axes_combination_layout->addWidget(_edit_axes_combination);
	right_layout->addLayout(custom_axes_combination_layout);

	connect(_custom_axis, SIGNAL(toggled(bool)), this, SLOT(show_edit_axes_widget(bool)));
	connect(_edit_axes_combination, SIGNAL(clicked()), this, SLOT(show_axes_combination_widget()));

	// TODO : add an OK button
	_axes_combination_widget = new PVAxesCombinationWidget(custom_axes_combination, &view);
	_axes_combination_widget->setWindowModality(Qt::WindowModal);
	_axes_combination_widget->hide();
}

void PVGuiQt::PVExportSelectionDlg::show_edit_axes_widget(bool show)
{
	_edit_axes_combination->setEnabled(show);
}

void PVGuiQt::PVExportSelectionDlg::show_axes_combination_widget()
{
	_axes_combination_widget->show();

}

PVGuiQt::PVExportSelectionDlg::AxisCombinationKind
PVGuiQt::PVExportSelectionDlg::combination_kind() const
{
	if(_all_axis->isChecked()) {
		return AxisCombinationKind::ALL;
	}
	else if(_current_axis->isChecked()) {
		return AxisCombinationKind::CURRENT;
	}
	else {
		assert(_custom_axis->isChecked());
		return AxisCombinationKind::CUSTOM;
	}
}

/* This is the static function used to export a selection.
 *
 * It creates the ExportSelectionDlg and handle the result.
 */
void PVGuiQt::PVExportSelectionDlg::export_selection(
	Picviz::PVView& view,
	const Picviz::PVSelection& sel)
{
	// Axis (column) to export
	Picviz::PVAxesCombination const& axes_combination = view.get_axes_combination();
	// Keep a separate value for custom axis selection
	Picviz::PVAxesCombination custom_axes_combination = axes_combination;

	// FileDialog for option selection and file to write
	PVGuiQt::PVExportSelectionDlg export_selection_dlg(custom_axes_combination, view);

	QFile file;
	// Ask for file until a valid name is given or the action is aborted
	while (true) {
		int res = export_selection_dlg.exec();
		QString filename = export_selection_dlg.selectedFiles()[0];
		if (filename.isEmpty() || res == QDialog::Rejected) {
			return;
		}

		file.setFileName(filename);
		if (file.open(QIODevice::WriteOnly)) {
			break;
		}

		// Error case
		QMessageBox::critical(
			&export_selection_dlg,
			tr("Error while exporting the selection"),
			tr("Can not create the file \"%1\"").arg(filename)
			);
	}

	// TODO: put an option in the widget for the file locale
	// Open a text stream with the current locale (by default in QTextStream)
	QTextStream stream(&file);

	// Get export characters parameters
	const QString sep_char = export_selection_dlg.separator_char();
	const QString quote_char = export_selection_dlg.quote_char();

	// Select the correct selection:
	// CUSTOM: use the export_selection_dlg axis combination and export
	// only selected axis
	// CURRENT: Same as custom but use the view axis combination
	// ALL: Use the view combination but export original axis
	PVCore::PVColumnIndexes column_indexes;
	QStringList str_list;
	switch(export_selection_dlg.combination_kind())
	{
		case AxisCombinationKind::CUSTOM:
			for(const Picviz::PVAxesCombination::axes_comb_id_t& a: custom_axes_combination.get_axes_index_list())
				column_indexes.push_back(a.get_axis());
			str_list = custom_axes_combination.get_axes_names_list();
			break;
		case AxisCombinationKind::CURRENT:
			for(const Picviz::PVAxesCombination::axes_comb_id_t& a: axes_combination.get_axes_index_list())
				column_indexes.push_back(a.get_axis());
			str_list = axes_combination.get_axes_names_list();
			break;
		case AxisCombinationKind::ALL:
			for(int a=0; a<axes_combination.get_original_axes_count(); a++)
				column_indexes.push_back(a);
			str_list = axes_combination.get_original_axes_names_list();
			break;
	}

	// Export header
	if (export_selection_dlg.export_columns_header()) {
		PVRush::PVUtils::safe_export(str_list, sep_char, quote_char);
		stream << "#" << str_list.join(sep_char) << "\n";
	}

	// Rows to export
	PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
	PVRow nrows = nraw.get_number_rows();

	PVRow start = 0;
	PVRow step_count = 20000;

	// Progress Bar for export advancement
	PVCore::PVProgressBox pbox("Selection export");

	// Export selected lines
	// TODO : We know the number of line to set a progression
	PVCore::PVProgressBox::progress([&]() {
		while (true) {
			start = sel.find_next_set_bit(start, nrows);
			if (start == PVROW_INVALID_VALUE) {
				break;
			}
			step_count = std::min(step_count, nrows - start);
			nraw.export_lines(stream, sel, column_indexes, start, step_count, sep_char, quote_char);
			start += step_count;
			if (pbox.get_cancel_state() != PVCore::PVProgressBox::CONTINUE) {
				return;
			}
		}
	}, &pbox);
}
