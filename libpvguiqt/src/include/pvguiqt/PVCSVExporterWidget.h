/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVGUIQT_PVCSVEXPORTERWIDGET_H__
#define __BPVGUIQT_PVCSVEXPORTERWIDGET_H__

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/widgets/PVExporterWidgetInterface.h>
#include <pvkernel/widgets/qkeysequencewidget.h>
#include <inendi/PVAxesCombination.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

namespace PVGuiQt
{

class PVCSVExporterWidget : public PVWidgets::PVExporterWidgetInterface
{
  public:
	PVCSVExporterWidget(Inendi::PVView const& view)
	    : _exporter(PVRush::PVCSVExporter()), _custom_axes_combination(view.get_axes_combination())
	{
		// Layout for export_layout is:
		// --------------------export_layout---------------------------
		// |----------------------------------------------------------|
		// ||  left_layout              |   right layout             ||
		// |----------------------------------------------------------|
		// ------------------------------------------------------------
		QHBoxLayout* export_layout = new QHBoxLayout();
		QVBoxLayout* left_layout = new QVBoxLayout();
		QVBoxLayout* right_layout = new QVBoxLayout();
		export_layout->addLayout(left_layout);
		export_layout->addLayout(right_layout);

		/// left_layout

		// Export column name
		QCheckBox* export_header = new QCheckBox("Export column names as header");
		export_header->setChecked(_exporter.get_export_header());

		export_header->setCheckState(Qt::CheckState::Checked);
		left_layout->addWidget(export_header);

		// Define csv specific character
		QFormLayout* char_layout = new QFormLayout();
		char_layout->setFieldGrowthPolicy(QFormLayout::FieldsStayAtSizeHint);

		// Separator character
		PVWidgets::QKeySequenceWidget* separator_char = new PVWidgets::QKeySequenceWidget();
		separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
		separator_char->setKeySequence(QKeySequence(","));
		separator_char->setMaxNumKey(1);
		char_layout->addRow("Fields separator:", separator_char);

		// Quote character
		PVWidgets::QKeySequenceWidget* quote_char = new PVWidgets::QKeySequenceWidget();
		quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
		quote_char->setKeySequence(QKeySequence("\""));
		quote_char->setMaxNumKey(1);
		char_layout->addRow("Quote character:", quote_char);

		left_layout->addLayout(char_layout);

		QCheckBox* export_internal_values =
		    new QCheckBox("Export internal values instead of displayed values");
		export_internal_values->setChecked(_exporter.get_export_internal_values());
		QObject::connect(export_internal_values, &QCheckBox::stateChanged,
		                 [&](int state) { _exporter.set_export_internal_values((bool)state); });
		left_layout->addWidget(export_internal_values);

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
		right_layout->addStretch();

		// TODO : add an OK button
		PVGuiQt::PVAxesCombinationWidget* axes_combination_widget =
		    new PVGuiQt::PVAxesCombinationWidget(_custom_axes_combination);
		axes_combination_widget->setWindowModality(Qt::WindowModal);
		axes_combination_widget->hide();

		QObject::connect(custom_axis, &QRadioButton::toggled,
		                 [&](bool show) { edit_axes_combination->setEnabled(show); });
		QObject::connect(edit_axes_combination, &QPushButton::clicked,
		                 [&]() { axes_combination_widget->show(); });

		//////
		// Rows to export
		PVRush::PVNraw const& nraw = view.get_rushnraw_parent();
		_exporter.set_total_row_count(nraw.row_count());

		PVRush::PVCSVExporter::export_func_f export_func =
		    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
		if (_exporter.get_export_internal_values()) {
			const Inendi::PVPlotted& plotted = view.get_parent<Inendi::PVPlotted>();
			export_func = [&](PVRow row, const PVCore::PVColumnIndexes& cols,
			                  const std::string& sep, const std::string& quote) {

				return plotted.export_line(row, cols, sep, quote);
			};
		}
		_exporter.set_export_func(export_func);

		setLayout(export_layout);
	}

  public:
	PVRush::PVCSVExporter& exporter() override { return _exporter; }

  private:
	PVRush::PVCSVExporter _exporter;
	Inendi::PVAxesCombination _custom_axes_combination;
};

} // namespace PVGuiQt

#endif // __BPVGUIQT_PVCSVEXPORTERWIDGET_H__
