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
#include <inendi/PVLayerStack.h>
#include <inendi/PVSource.h>

#include <pvkernel/core/PVProgressBox.h>

#include <filesystem>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(
	Inendi::PVView& view,
	QWidget* parent /* = 0 */,
	QFileDialog::AcceptMode accept_mode /* = QFileDialog::AcceptSave */,
	QFileDialog::FileMode file_mode /* = QFileDialog::AnyFile */)
    : PVWidgets::PVExportDlg(parent, accept_mode, file_mode)
{
	delete _groupbox->layout();
	delete _exporter_widget;

	// Use default CSV exporter if input type doesn't define a specific one
	PVGuiQt::PVCSVExporterWidget* exporter_widget = new PVGuiQt::PVCSVExporterWidget(view);
	_exporter = &exporter_widget->exporter();

	// Add input specific exporter filter string if any
	const Inendi::PVSource& source = view.get_parent<Inendi::PVSource>();
	const QString& specific_export_filter =
	    source.get_source_creator()->supported_type_lib()->get_exporter_filter_string(
	        source.get_inputs());
	if (not specific_export_filter.isEmpty()) {
		_name_filters.insert(0, specific_export_filter);
	}

	setNameFilters(_name_filters);

	// Set specific exporter as default if any, .csv.gz otherwise
	int gz_idx;
	for (gz_idx = _name_filters.size() - 1; gz_idx >= 0; gz_idx--) {
		if (_name_filters[gz_idx].contains("gz")) {
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
	_groupbox->setLayout(stacked_layout);

	auto filter_selected_f = [=,this](const QString& filter) {
		_filter_selected_f(filter);

		_is_source_exporter = not filter.contains("*.csv");
		stacked_layout->setCurrentIndex(_is_source_exporter);
	};
	QObject::disconnect(_conn);
	QObject::connect(this, &QFileDialog::filterSelected, filter_selected_f);
	filter_selected_f(_name_filters.at(default_filter_index));
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

void PVGuiQt::PVExportSelectionDlg::export_layers(Inendi::PVView& view)
{
	const Inendi::PVLayerStack& layerstack = view.get_layer_stack();

	PVGuiQt::PVExportSelectionDlg export_dialog(view, nullptr, QFileDialog::AcceptOpen, QFileDialog::Directory);
	int res = export_dialog.exec();
	QString dirname = export_dialog.selectedFiles()[0];
	if (dirname.isEmpty() || res == QDialog::Rejected) {
		return;
	}

	size_t total_row_count = 0;
	for (int i = 0; i < layerstack.get_layer_count(); i++) {
		total_row_count += layerstack.get_layer_n(i).get_selectable_count();
	}

	PVRush::PVExporterBase& exporter = export_dialog.exporter();

	// Check if some files are being overwritten
	std::vector<std::string> files_path;
	bool overwritten = false;
	for (int i = 0; i < layerstack.get_layer_count(); i++) {
		files_path.emplace_back((dirname + "/" + layerstack.get_layer_n(i).get_name() + export_dialog.file_extension()).toStdString());
		overwritten |= std::filesystem::exists(files_path.back());
	}
	if (overwritten) {
		if (QMessageBox::warning(
			nullptr,
			"Overwrite files ?",
			"Some files are being overwritten, do you want to continue ?",
			QMessageBox::Ok | QMessageBox::Cancel) == QMessageBox::Cancel) {
				return;
			}
	}

	// Export selected lines
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    // setup progression tracking
		    exporter.set_progress_max_func(
		        [&](size_t row_count) { pbox.set_maximum(row_count); },
		        total_row_count);
		    exporter.set_progress_func([&](size_t exported_row_count) {
			    if (pbox.get_cancel_state() != PVCore::PVProgressBox::CancelState::CONTINUE) {
				    exporter.cancel();
				    return;
			    }
			    pbox.set_value(exported_row_count);
		    });

			for (int i = 0; i < layerstack.get_layer_count(); i++) {
				const Inendi::PVLayer& layer = layerstack.get_layer_n(i);
				const std::string& file_path = files_path[i];
				try {
					exporter.export_rows(file_path, layer.get_selection());
				} catch (const PVRush::PVExportError& e) {
					pbox.critical("Error when exporting data", e.what());
					std::remove(file_path.c_str());
				}
			}
	    },
	    "Export layers", nullptr);
}
