/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSpacerItem>
#include <QVBoxLayout>

#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <inendi/PVSelection.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>
#include <inendi/PVPlotted.h>

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/PVProgressBox.h>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(
    Inendi::PVAxesCombination& custom_axes_combination,
    Inendi::PVView& view,
    QWidget* parent /* = 0 */)
    : QFileDialog(parent)
{
	// Set this flags to make sure we can access the layout.
	setOption(QFileDialog::DontUseNativeDialog);

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
	QGridLayout* main_layout = static_cast<QGridLayout*>(layout());

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

	// Add input specific exporter filter string if any
	QStringList name_filters;
	const Inendi::PVSource& source = view.get_parent<Inendi::PVSource>();
	const QString& specific_export_filter =
	    source.get_source_creator()->supported_type_lib()->get_exporter_filter_string(
	        source.get_inputs());
	if (not specific_export_filter.isEmpty()) {
		name_filters << specific_export_filter;
	}

	// Default CSV exporter
	name_filters << ".csv files (*.csv)";
	for (const std::string& extension : PVCore::PVStreamingCompressor::supported_extensions()) {
		if (std::ifstream(PVCore::PVStreamingCompressor::executable(extension)).good()) {
			const QString& name_filter =
			    QString::fromStdString(".csv." + extension + " files (*.csv." + extension + ")");
			name_filters << name_filter;
		}
	}
	setNameFilters(name_filters);
	auto suffix_from_filter = [](const QString& filter) { return filter.split(" ")[0]; };

	// Set specific exporter as default if any, .csv.gz otherwise
	int gz_idx;
	for (gz_idx = name_filters.size() - 1; gz_idx >= 0; gz_idx--) {
		if (name_filters[gz_idx].contains("gz")) {
			break;
		}
	}
	int default_filter_index = specific_export_filter.isEmpty() ? gz_idx : 0;

	setDefaultSuffix(suffix_from_filter(name_filters.at(default_filter_index)));
	connect(this, &QFileDialog::filterSelected, [=](const QString& filter) {
		setDefaultSuffix(suffix_from_filter(filter));
		group_box->setVisible(filter.contains("*.csv"));
	});
	group_box->setVisible(specific_export_filter.isEmpty());

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

	_export_internal_values = new QCheckBox("Export internal values instead of displayed values");
	left_layout->addWidget(_export_internal_values);

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
	right_layout->addStretch();

	connect(_custom_axis, &QRadioButton::toggled, this,
	        &PVGuiQt::PVExportSelectionDlg::show_edit_axes_widget);
	connect(_edit_axes_combination, &QPushButton::clicked, this,
	        &PVGuiQt::PVExportSelectionDlg::show_axes_combination_widget);

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
	if (_all_axis->isChecked()) {
		return AxisCombinationKind::ALL;
	} else if (_current_axis->isChecked()) {
		return AxisCombinationKind::CURRENT;
	} else {
		assert(_custom_axis->isChecked());
		return AxisCombinationKind::CUSTOM;
	}
}

/* This is the static function used to export a selection.
 *
 * It creates the ExportSelectionDlg and handle the result.
 */
void PVGuiQt::PVExportSelectionDlg::export_selection(Inendi::PVView& view,
                                                     const Inendi::PVSelection& sel)
{
	// Axis (column) to export
	Inendi::PVAxesCombination const& axes_combination = view.get_axes_combination();
	// Keep a separate value for custom axis selection
	Inendi::PVAxesCombination custom_axes_combination = axes_combination;

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
		QMessageBox::critical(&export_selection_dlg, tr("Error while exporting the selection"),
		                      tr("Can not create the file \"%1\"").arg(filename));
	}

	// Get export characters parameters
	const std::string sep_char = export_selection_dlg.separator_char().toStdString();
	const std::string quote_char = export_selection_dlg.quote_char().toStdString();

	// Select the correct selection:
	// CUSTOM: use the export_selection_dlg axis combination and export
	// only selected axis
	// CURRENT: Same as custom but use the view axis combination
	// ALL: Use the view combination but export original axis
	PVCore::PVColumnIndexes column_indexes;
	QStringList str_list;
	switch (export_selection_dlg.combination_kind()) {
	case AxisCombinationKind::CUSTOM:
		column_indexes = custom_axes_combination.get_combination();
		str_list = custom_axes_combination.get_combined_names();
		break;
	case AxisCombinationKind::CURRENT:
		column_indexes = axes_combination.get_combination();
		str_list = axes_combination.get_combined_names();
		break;
	case AxisCombinationKind::ALL:
		for (PVCol a(0); a < view.get_parent<Inendi::PVSource>().get_nraw_column_count(); a++)
			column_indexes.push_back(a);
		str_list = axes_combination.get_nraw_names();
		break;
	}

	// Export header
	std::string header;
	if (export_selection_dlg.export_columns_header()) {
		PVRush::PVUtils::safe_export(str_list, sep_char, quote_char);
		header = "#" + str_list.join(export_selection_dlg.separator_char()).toStdString() + "\n";
	}

	// Rows to export
	PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
	PVRow nrows = nraw.row_count();

	// Progress Bar for export advancement
	bool export_internal_values = export_selection_dlg._export_internal_values->isChecked();

	PVCore::PVCSVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
	if (export_internal_values) {
		const Inendi::PVPlotted& plotted = view.get_parent<Inendi::PVPlotted>();
		export_func = [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		                  const std::string& quote) {

			return plotted.export_line(row, cols, sep, quote);
		};
	}

	const Inendi::PVSource& source = view.get_parent<Inendi::PVSource>();
	std::unique_ptr<PVCore::PVExporterBase> exp =
	    source.get_source_creator()->supported_type_lib()->create_exporter(
	        file.fileName().toStdString(), sel, source.get_inputs(), nraw);

	// Use default CSV exporter if input type doesn't define a specific one
	if (export_selection_dlg.selectedNameFilter().contains("*.csv")) {
		exp.reset(new PVCore::PVCSVExporter(file.fileName().toStdString(), sel, column_indexes,
		                                    nrows, export_func, sep_char, quote_char, header));
	}

	// Export selected lines
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {

		    // setup progression tracking
		    exp->set_progress_max_func(
		        [&](size_t total_row_count) { pbox.set_maximum(total_row_count); });
		    exp->set_progress_func([&](size_t exported_row_count) {
			    if (pbox.get_cancel_state() != PVCore::PVProgressBox::CancelState::CONTINUE) {
				    exp->cancel();
				    return;
			    }
			    pbox.set_value(exported_row_count);
			});

		    try {
			    exp->export_rows();
		    } catch (const PVCore::PVExportError& e) {
			    pbox.critical("Error when exporting data", e.what());
			    std::remove(file.fileName().toStdString().c_str());
		    }
		},
	    "Selection export", nullptr);
}
