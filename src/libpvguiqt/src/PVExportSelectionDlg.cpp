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

#include <QFile>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSpacerItem>
#include <QVBoxLayout>

#include <pvguiqt/PVAxesCombinationExportWidget.h>
#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVCSVExporterWidget.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include <squey/PVSelection.h>
#include <squey/PVView.h>
#include <squey/PVLayerStack.h>
#include <squey/PVSource.h>

#include <pvkernel/core/PVProgressBox.h>

#include <filesystem>
#include <utility>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(Squey::PVView& view,
                                                    QWidget* parent /* = 0 */,
                                                    Target target /* = Target::File */)
    : PVWidgets::PVExportDlg(parent, target, new PVGuiQt::PVCSVExporterWidget(view))
{
	setObjectName("PVExportSelectionDlg");

	const auto add_page = [&](const QString& filter, PVWidgets::PVExporterWidgetInterface* widget,
	                          PVAxesCombinationExportWidget::Choice choice,
	                          const QString& all_label) {
		if (widget == nullptr) {
			return;
		}
		PVRush::PVExporterBase* exporter = &widget->exporter();
		QWidget* page = widget;
		if (exporter->exports_columns()) {
			page = new QWidget;
			auto* layout = new QVBoxLayout(page);
			layout->setContentsMargins(0, 0, 0, 0);
			layout->addWidget(widget);
			auto* axes = new PVAxesCombinationExportWidget(view, choice, all_label);
			layout->addWidget(axes);
			layout->addStretch();
			// No column chosen exports them all: the input as it is, for the exporter of its type.
			const auto export_axes = [exporter, axes]() {
				exporter->set_columns(axes->choice() == PVAxesCombinationExportWidget::Choice::All
				                          ? PVCore::PVColumnIndexes()
				                          : axes->column_indexes());
			};
			QObject::connect(axes, &PVAxesCombinationExportWidget::changed, widget, export_axes);
			export_axes();
		}
		add_format(filter, page, *exporter);
	};

	const Squey::PVSource& source = view.get_parent<Squey::PVSource>();
	const QString source_type_name = source.get_source_creator()->supported_type();
	const PVRush::PVInputType_p source_type = source.get_source_creator()->supported_type_lib();
	PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
	const QString source_filter = source_type->get_exporter_filter_string(source.get_inputs());
	add_page(source_filter, source_type->create_exporter_widget(source.get_inputs(), nraw),
	         PVAxesCombinationExportWidget::Choice::All, tr("Use all the columns of the file"));
	const bool has_source_exporter = name_filters().contains(source_filter);

	const QStringList column_names = view.get_axes_combination().get_nraw_names();
	for (const auto& type : LIB_CLASS(PVRush::PVInputType)::get().get_list()) {
		const QString filter = type.value()->get_nraw_exporter_filter_string();
		// That of the type of the source too, if this type has no exporter for these inputs.
		if ((type.key() != source_type_name or not has_source_exporter) and not filter.isEmpty() and
		    not name_filters().contains(filter)) {
			add_page(filter, type.value()->create_nraw_exporter_widget(nraw, column_names),
			         PVAxesCombinationExportWidget::Choice::Current,
			         tr("Use all axes combination"));
		}
	}

	// Set specific exporter as default if any, otherwise select default platform compressor
	if (has_source_exporter) {
		select_name_filter(source_filter);
	} else {
#ifdef __linux__
		const QString default_compressor = "gz";
#else
		const QString default_compressor = "zip";
#endif
		const QStringList filters = name_filters();
		for (qsizetype index = filters.size() - 1; index >= 0; index--) {
			if (filters[index].contains(default_compressor)) {
				select_name_filter(filters[index]);
				break;
			}
		}
	}
}

/* This is the static function used to export a selection.
 *
 * It creates the ExportSelectionDlg and handle the result.
 */
void PVGuiQt::PVExportSelectionDlg::export_selection(Squey::PVView& view,
                                                     const Squey::PVSelection& sel,
													 QWidget* parent /* = nullptr */)
{
	PVGuiQt::PVExportSelectionDlg export_selection_dlg(view, parent);

	QFile file;
	// Ask for file until a valid name is given or the action is aborted
	while (true) {
		if (export_selection_dlg.exec() != QDialog::Accepted) {
			return;
		}
		const QString filename = export_selection_dlg.path();

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
		    } catch (const std::exception& e) {
			    pbox.critical("Error when exporting data", e.what());
			    std::remove(file.fileName().toStdString().c_str());
		    }
	    },
	    "Selection export", nullptr);

	Q_EMIT export_selection_dlg.selection_exported();
}

void PVGuiQt::PVExportSelectionDlg::export_layers(Squey::PVView& view)
{
	const Squey::PVLayerStack& layerstack = view.get_layer_stack();

	PVGuiQt::PVExportSelectionDlg export_dialog(view, nullptr, Target::Directory);
	if (export_dialog.exec() != QDialog::Accepted) {
		return;
	}
	const QString dirname = export_dialog.path();

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
		overwritten |= QFile::exists(QString::fromStdString(files_path.back()));
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
				const Squey::PVLayer& layer = layerstack.get_layer_n(i);
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
