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

#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>

#include <inendi/widgets/PVMappingModeWidget.h>
#include <inendi/widgets/PVPlottingModeWidget.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QScrollArea>

static QList<PVRush::PVAxisFormat> const& init_axes(Inendi::PVMapped* mapping,
                                                    Inendi::PVPlotted* plotting)
{
	if (mapping) {
		return mapping->get_parent<Inendi::PVSource>().get_format().get_axes();
	}

	assert(plotting);
	return plotting->get_parent<Inendi::PVSource>().get_format().get_axes();
}

/******************************************************************************
 *
 * Inendi::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog
 *
 *****************************************************************************/
PVWidgets::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog(Inendi::PVMapped* mapping,
                                                                    Inendi::PVPlotted* plotting,
                                                                    QWidget* parent)
    : QDialog(parent), _mapping(mapping), _plotting(plotting), _axes(init_axes(_mapping, _plotting))
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

#ifndef NDEBUG
	if (has_mapping() && has_plotting()) {
		assert(&_mapping->get_parent<Inendi::PVSource>() ==
		       &_plotting->get_parent<Inendi::PVSource>());
	} else {
		assert(has_mapping() || has_plotting());
	}
#endif

	setWindowTitle(tr("Edit properties..."));

	// We generate and populate the Layout
	init_layout();
	// We load the current settings
	load_settings();
	// We can set the layout in the Widget
	finish_layout();

	// Adapt scroll area width so that everything fits in (closes #0x100)
	_main_group_box->setMinimumWidth(_main_scroll_area->viewport()->width());
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::create_label
 *
 *****************************************************************************/
QLabel* PVWidgets::PVMappingPlottingEditDialog::create_label(QString const& text,
                                                             Qt::Alignment align)
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	QLabel* ret = new QLabel(text, nullptr);
	ret->setAlignment(align);
	QFont font(ret->font());
	font.setBold(true);
	ret->setFont(font);
	return ret;
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::finish_layout
 *
 *****************************************************************************/
void PVWidgets::PVMappingPlottingEditDialog::finish_layout()
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(btns, &QDialogButtonBox::accepted, this, &PVMappingPlottingEditDialog::save_settings);
	connect(btns, &QDialogButtonBox::rejected, this, &QDialog::reject);
	_main_layout->addWidget(btns);
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::init_layout
 *
 *****************************************************************************/
void PVWidgets::PVMappingPlottingEditDialog::init_layout()
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	_main_layout = new QVBoxLayout();
	_main_layout->setSpacing(29);

	QHBoxLayout* name_layout = new QHBoxLayout();
	name_layout->addWidget(new QLabel(tr("Name:"), nullptr));
	_edit_name = new QLineEdit();
	name_layout->addWidget(_edit_name);
	_main_layout->addLayout(name_layout);

	QWidget* grid_widget = new QWidget();
	_main_grid = new QGridLayout(grid_widget);
	_main_grid->setContentsMargins(0, 0, 0, 0);
	_main_grid->setHorizontalSpacing(10);
	_main_grid->setVerticalSpacing(10);
	int row = 0;
	int col = 0;

	// Init titles
	_main_grid->addWidget(create_label(tr("Axis"), Qt::AlignLeft), row, col);
	_main_grid->setColumnStretch(col, 1);
	col++;
	if (has_mapping()) {
		_main_grid->addWidget(create_label(tr("Type")), row, col);
		_main_grid->setColumnStretch(col, 1);
		col++;
		_main_grid->addWidget(create_label(tr("Mapping")), row, col);
		_main_grid->setColumnStretch(col, 2);
		col++;
	}
	if (has_plotting()) {
		_main_grid->addWidget(create_label(tr("Plotting")), row, col);
		_main_grid->setColumnStretch(col, 2);
		col++;
	}
	_main_scroll_area = new QScrollArea();
	_main_scroll_area->setWidget(grid_widget);
	_main_scroll_area->setWidgetResizable(true);

	QVBoxLayout* scroll_layout = new QVBoxLayout();
	scroll_layout->addWidget(_main_scroll_area);

	_main_group_box = new QGroupBox(tr("Parameters"));
	_main_group_box->setSizePolicy(
	    QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	_main_group_box->setLayout(scroll_layout);
	_main_layout->addWidget(_main_group_box);

	setLayout(_main_layout);
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::load_settings
 *
 *****************************************************************************/
void PVWidgets::PVMappingPlottingEditDialog::load_settings()
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	// We must get the official name of the
	QString name;
	if (has_mapping()) {
		name = QString::fromStdString(_mapping->get_name());
	} else {
		name = QString::fromStdString(_plotting->get_name());
	}
	_edit_name->setText(name);

	// Add widgets

	int row = 1;
	for (PVRush::PVAxisFormat const& axe : _axes) {
		int col = 0;
		_main_grid->addWidget(new QLabel(axe.get_name(), this), row, col++);
		if (has_mapping()) {
			_main_grid->addWidget(new QLabel(axe.get_type()), row, col++);
			_main_grid->addWidget(new PVWidgets::PVMappingModeWidget(axe.index, *_mapping, this),
			                      row, col++);
		}
		if (has_plotting()) {
			_main_grid->addWidget(new PVWidgets::PVPlottingModeWidget(axe.index, *_plotting, this),
			                      row, col++);
		}
		row++;
	}
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::save_settings
 *
 *****************************************************************************/
void PVWidgets::PVMappingPlottingEditDialog::save_settings()
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	QString name = _edit_name->text();
	if (name.isEmpty()) {
		_edit_name->setFocus(Qt::MouseFocusReason);
		return;
	}

	// If we're editing both at the same time, give the same name !
	if (has_mapping()) {
		_mapping->set_name(name.toStdString());
	}
	if (has_plotting()) {
		_plotting->set_name(name.toStdString());
	}

	int row = 1;
	for (PVRush::PVAxisFormat const& axis : _axes) {
		if (has_mapping()) {
			QString type = axis.get_type();
			Inendi::PVMappingProperties& prop = _mapping->get_properties_for_col(axis.index);

			// Mapping mode
			PVWidgets::PVMappingModeWidget* map_combo =
			    dynamic_cast<PVWidgets::PVMappingModeWidget*>(
			        _main_grid->itemAtPosition(row, 2)->widget());
			assert(map_combo);
			QString mode = map_combo->get_mode();

			prop.set_mode(mode.toStdString());
		}
		if (has_plotting()) {
			PVWidgets::PVPlottingModeWidget* combo = dynamic_cast<PVWidgets::PVPlottingModeWidget*>(
			    _main_grid->itemAtPosition(row, 1)->widget());
			assert(combo);
			QString mode = combo->get_mode();
			_plotting->get_properties_for_col(axis.index).set_mode(mode.toStdString());
		}
		row++;
	}

	accept();
}
