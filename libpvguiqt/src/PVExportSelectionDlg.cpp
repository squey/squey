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
#include <QStackedLayout>

#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVCSVExporterWidget.h>

#include <inendi/PVSelection.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>

#include <pvkernel/core/PVProgressBox.h>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(Inendi::PVView& view, QWidget* parent /* = 0 */)
    : PVWidgets::PVFileDialog(parent)
{
	// Set this flags to make sure we can access the layout.
	setOption(QFileDialog::DontUseNativeDialog);
	setWindowTitle(tr("Export selection"));
	setAcceptMode(QFileDialog::AcceptSave);

	// Use default CSV exporter if input type doesn't define a specific one
	PVGuiQt::PVCSVExporterWidget* exporter_widget = new PVGuiQt::PVCSVExporterWidget(view);
	_exporter = &exporter_widget->exporter();

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
		const QString& name_filter =
		    QString::fromStdString(".csv." + extension + " files (*.csv." + extension + ")");
		name_filters << name_filter;
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

	QStackedLayout* stacked_layout = new QStackedLayout;
	stacked_layout->addWidget(exporter_widget);
	PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
	PVWidgets::PVExporterWidgetInterface* specific_export_widget =
	    source.get_source_creator()->supported_type_lib()->create_exporter_widget(
	        source.get_inputs(), nraw);
	if (specific_export_widget) {
		_source_exporter = &specific_export_widget->exporter();
		stacked_layout->addWidget(specific_export_widget);
		stacked_layout->setCurrentIndex(1);
	}
	group_box->setLayout(stacked_layout);

	setDefaultSuffix(suffix_from_filter(name_filters.at(default_filter_index)));
	_is_source_exporter = not name_filters.at(default_filter_index).contains("*.csv");
	QObject::connect(this, &QFileDialog::filterSelected, [=](const QString& filter) {
		setDefaultSuffix(suffix_from_filter(filter));
		_is_source_exporter = not filter.contains("*.csv");
		stacked_layout->setCurrentIndex(_is_source_exporter);

		// force filters to reset as setting 'QFileDialog::Directory' erase them
		setNameFilters(name_filters);
		selectNameFilter(filter);
	});
}

/* This is the static function used to export a selection.
 *
 * It creates the ExportSelectionDlg and handle the result.
 */
void PVGuiQt::PVExportSelectionDlg::export_selection(Inendi::PVView& view,
                                                     const Inendi::PVSelection& sel)
{
	// FileDialog for option selection and file to write
	PVGuiQt::PVExportSelectionDlg export_selection_dlg(view);

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

	PVRush::PVExporterBase& exporter = export_selection_dlg.exporter();

	// Export selected lines
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    // setup progression tracking
		    exporter.set_progress_max_func(
		        [&](size_t total_row_count) { pbox.set_maximum(total_row_count); },
		        sel.bit_count());
		    exporter.set_progress_func([&](size_t exported_row_count) {
			    if (pbox.get_cancel_state() != PVCore::PVProgressBox::CancelState::CONTINUE) {
				    exporter.cancel();
				    return;
			    }
			    pbox.set_value(exported_row_count);
			});

		    try {
			    exporter.export_rows(file.fileName().toStdString(), sel);
		    } catch (const PVRush::PVExportError& e) {
			    pbox.critical("Error when exporting data", e.what());
			    std::remove(file.fileName().toStdString().c_str());
		    }
		},
	    "Selection export", nullptr);
}
