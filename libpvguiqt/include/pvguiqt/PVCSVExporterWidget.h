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
#define __BPVGUIQT_PVCSVEXPORTERWIDGET_H__

#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/widgets/PVCSVExporterWidget.h>
#include <pvkernel/widgets/qkeysequencewidget.h>
#include <inendi/PVAxesCombination.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

#include <QButtonGroup>

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

		// left_layout
		QCheckBox* export_rows_index = new QCheckBox("Export rows index");
		QObject::connect(export_rows_index, &QCheckBox::stateChanged, [&](int state) {
			_exporter.set_export_rows_index(state);
			if (_selected_radio_button) { // force header regeneration
				Q_EMIT _selected_radio_button->toggled(_selected_radio_button->isChecked());
			}
		});

		export_rows_index->setCheckState(Qt::CheckState::Unchecked);
		((QVBoxLayout*)_export_layout->itemAt(0)->layout())->insertWidget(1, export_rows_index);

		/// right_layout
		QButtonGroup* button_group = new QButtonGroup(this);

		// Use all axes combination
		QRadioButton* all_axis = new QRadioButton("Use all axes combination");
		button_group->addButton(all_axis);
		QObject::connect(all_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				PVCore::PVColumnIndexes column_indexes;
				for (PVCol a(0); a < view.get_parent<Inendi::PVSource>().get_nraw_column_count();
				     a++) {
					column_indexes.push_back(a);
				}
				_exporter.set_column_indexes(column_indexes);
				_exporter.set_header(view.get_axes_combination().get_nraw_names());
				_selected_radio_button = all_axis;
			}
		});
		right_layout->addWidget(all_axis);

		// Use current axes combination
		QRadioButton* current_axis = new QRadioButton("Use current axes combination");
		button_group->addButton(current_axis);
		QObject::connect(current_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				_exporter.set_column_indexes(view.get_axes_combination().get_combination());
				_exporter.set_header(view.get_axes_combination().get_combined_names());
				_selected_radio_button = current_axis;
			}
		});
		current_axis->setChecked(true);
		right_layout->addWidget(current_axis);

		// Use custom axes combination
		QHBoxLayout* custom_axes_combination_layout = new QHBoxLayout();
		QRadioButton* custom_axis = new QRadioButton("Use custom axes combination");
		button_group->addButton(custom_axis);
		QObject::connect(custom_axis, &QRadioButton::toggled, [&](bool checked) {
			if (checked) {
				_exporter.set_column_indexes(_custom_axes_combination.get_combination());
				_exporter.set_header(_custom_axes_combination.get_combined_names());
				_selected_radio_button = custom_axis;
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

		QObject::connect(
		    this, &PVWidgets::PVCSVExporterWidget::separator_char_changed,
		    [&, button_group]() { Q_EMIT button_group->checkedButton()->toggled(true); });

		QObject::connect(
		    this, &PVWidgets::PVCSVExporterWidget::quote_char_changed,
		    [&, button_group]() { Q_EMIT button_group->checkedButton()->toggled(true); });

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
			    std::string exported_row;
			    if (_exporter.get_export_rows_index()) {
				    exported_row = std::to_string(row + 1) + sep;
			    }
			    if (_exporter.get_export_internal_values()) {
				    exported_row += plotted.export_line(row, cols, sep, quote);
			    } else {
				    exported_row += nraw.export_line(row, cols, sep, quote);
			    }
			    return exported_row;
		    };
		_exporter.set_export_func(export_func);
	}

  public:
	PVRush::PVCSVExporter& exporter() override { return _exporter; }

  private:
	Inendi::PVAxesCombination _custom_axes_combination;
	QRadioButton* _selected_radio_button = nullptr;
};

} // namespace PVGuiQt

#endif // __BPVGUIQT_PVCSVEXPORTERWIDGET_H__
