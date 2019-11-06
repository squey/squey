/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVCSVEXPORTERWIDGET_H__
#define __BPVGUIQT_PVCSVEXPORTERWIDGET_H__

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>
#include <pvkernel/widgets/qkeysequencewidget.h>
#include <inendi/PVAxesCombination.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

namespace PVGuiQt
{

class PVCSVExporterWidget : public PVWidgets::PVCSVExporterWidget
{
  public:
	PVCSVExporterWidget(Inendi::PVView const& view)
	    : PVWidgets::PVCSVExporterWidget(), _custom_axes_combination(view.get_axes_combination())
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

		// Use all axes combination
		QRadioButton* all_axis = new QRadioButton("Use all axes combination");
		QObject::connect(all_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				PVCore::PVColumnIndexes column_indexes;
				for (PVCol a(0); a < view.get_parent<Inendi::PVSource>().get_nraw_column_count();
				     a++) {
					column_indexes.push_back(a);
				}
				_exporter.set_column_indexes(column_indexes);
				_exporter.set_header(view.get_axes_combination().get_nraw_names());
			}
		});
		right_layout->addWidget(all_axis);

		// Use current axes combination
		QRadioButton* current_axis = new QRadioButton("Use current axes combination");
		QObject::connect(current_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				_exporter.set_column_indexes(view.get_axes_combination().get_combination());
				_exporter.set_header(view.get_axes_combination().get_combined_names());
			}
		});
		current_axis->setChecked(true);
		right_layout->addWidget(current_axis);

		// Use custom axes combination
		QHBoxLayout* custom_axes_combination_layout = new QHBoxLayout();
		QRadioButton* custom_axis = new QRadioButton("Use custom axes combination");
		QObject::connect(custom_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				_exporter.set_column_indexes(_custom_axes_combination.get_combination());
				_exporter.set_header(_custom_axes_combination.get_combined_names());
			}
		});
		QPushButton* edit_axes_combination = new QPushButton("Edit");
		edit_axes_combination->setEnabled(custom_axis->isChecked());

		custom_axes_combination_layout->addWidget(custom_axis);
		custom_axes_combination_layout->addWidget(edit_axes_combination);
		right_layout->addLayout(custom_axes_combination_layout);

		// Export internal values
		QCheckBox* export_internal_values =
		    new QCheckBox("Export internal values instead of displayed values");
		export_internal_values->setChecked(_exporter.get_export_internal_values());
		QObject::connect(export_internal_values, &QCheckBox::stateChanged,
		                 [&](int state) { _exporter.set_export_internal_values((bool)state); });
		right_layout->addWidget(export_internal_values);

		right_layout->addStretch();

		// TODO : add an OK button
		PVGuiQt::PVAxesCombinationWidget* axes_combination_widget =
		    new PVGuiQt::PVAxesCombinationWidget(_custom_axes_combination);
		axes_combination_widget->setWindowModality(Qt::WindowModal);
		axes_combination_widget->hide();

		QObject::connect(custom_axis, &QRadioButton::toggled,
		                 [=](bool show) { edit_axes_combination->setEnabled(show); });
		QObject::connect(edit_axes_combination, &QPushButton::clicked,
		                 [=]() { axes_combination_widget->show(); });

		//////
		// Rows to export
		PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
		_exporter.set_total_row_count(nraw.row_count());
		const Inendi::PVPlotted& plotted = view.get_parent<Inendi::PVPlotted>();
		PVRush::PVCSVExporter::export_func_f export_func =
		    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		        const std::string& quote) {
			    if (_exporter.get_export_internal_values()) {
				    return plotted.export_line(row, cols, sep, quote);
			    } else {
				    return nraw.export_line(row, cols, sep, quote);
			    }
		    };
		_exporter.set_export_func(export_func);
	}

  public:
	PVRush::PVCSVExporter& exporter() override { return _exporter; }

  private:
	PVRush::PVCSVExporter _exporter;
	Inendi::PVAxesCombination _custom_axes_combination;
};

} // namespace PVGuiQt

#endif // __BPVGUIQT_PVCSVEXPORTERWIDGET_H__
