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

#ifndef __PVGUIQT_PVCSVEXPORTERWIDGET_H__
#define __PVGUIQT_PVCSVEXPORTERWIDGET_H__

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>
#include <pvkernel/widgets/qkeysequencewidget.h>
#include <squey/PVView.h>
#include <squey/PVSource.h>
#include <pvguiqt/PVAxesCombinationExportWidget.h>

namespace PVGuiQt
{

class PVCSVExporterWidget : public PVWidgets::PVCSVExporterWidget
{
  public:
	PVCSVExporterWidget(Squey::PVView const& view) : PVWidgets::PVCSVExporterWidget()
	{
		// Layout for export_layout is:
		// --------------------export_layout---------------------------
		// |----------------------------------------------------------|
		// ||  left_layout              |   right layout             ||
		// |----------------------------------------------------------|
		// ------------------------------------------------------------
		QVBoxLayout* right_layout = new QVBoxLayout();
		_export_layout->addLayout(right_layout);

		/// right_layout
		auto* axes = new PVAxesCombinationExportWidget(
		    view, PVAxesCombinationExportWidget::Choice::Current, "Use all axes combination");
		right_layout->addWidget(axes);
		// The header names the row index, and is quoted and separated as the rows are.
		const auto export_axes = [this, axes]() {
			_exporter.set_column_indexes(axes->column_indexes());
			_exporter.set_header(axes->names());
		};
		QObject::connect(axes, &PVAxesCombinationExportWidget::changed, export_axes);
		QObject::connect(this, &PVWidgets::PVCSVExporterWidget::separator_char_changed,
		                 export_axes);
		QObject::connect(this, &PVWidgets::PVCSVExporterWidget::quote_char_changed, export_axes);
		export_axes();

		// left_layout
		QCheckBox* export_rows_index = new QCheckBox("Export rows index");
		QObject::connect(export_rows_index, &QCheckBox::checkStateChanged,
		                 [this, export_axes](Qt::CheckState state) {
			                 _exporter.set_export_rows_index(state == Qt::Checked);
			                 export_axes();
		                 });

		export_rows_index->setCheckState(Qt::CheckState::Unchecked);
		((QVBoxLayout*)_export_layout->itemAt(0)->layout())->insertWidget(1, export_rows_index);

		// Export internal values
		QCheckBox* export_internal_values =
		    new QCheckBox("Export internal values instead of displayed values");
		export_internal_values->setChecked(_exporter.get_export_internal_values());
		QObject::connect(export_internal_values, &QCheckBox::checkStateChanged,
		                 [&](int state) { _exporter.set_export_internal_values((bool)state); });
		right_layout->addWidget(export_internal_values);

		right_layout->addStretch();

		//////
		// Rows to export
		PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
		_exporter.set_total_row_count(nraw.row_count());
		const Squey::PVScaled& scaled = view.get_parent<Squey::PVScaled>();
		PVRush::PVCSVExporter::export_func_f export_func =
		    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		        const std::string& quote) {
			    std::string exported_row;
			    if (_exporter.get_export_rows_index()) {
				    exported_row = std::to_string(row + 1) + sep;
			    }
			    if (_exporter.get_export_internal_values()) {
				    exported_row += scaled.export_line(row, cols, sep, quote);
			    } else {
				    exported_row += nraw.export_line(row, cols, sep, quote);
			    }
			    return exported_row;
		    };
		_exporter.set_export_func(export_func);
	}

  public:
	PVRush::PVCSVExporter& exporter() override { return _exporter; }
};

} // namespace PVGuiQt

#endif // __PVGUIQT_PVCSVEXPORTERWIDGET_H__
