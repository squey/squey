/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapping.h>
#include <inendi/PVPlotting.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <inendi/widgets/PVAxisTypeWidget.h>
#include <inendi/widgets/PVMappingModeWidget.h>
#include <inendi/widgets/PVPlottingModeWidget.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QScrollBar>

/******************************************************************************
 *
 * Inendi::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog
 *
 *****************************************************************************/
PVWidgets::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog(Inendi::PVMapping* mapping,
                                                                    Inendi::PVPlotting* plotting,
                                                                    QWidget* parent)
    : QDialog(parent), _mapping(mapping), _plotting(plotting)
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

#ifndef NDEBUG
	if (has_mapping() && has_plotting()) {
		assert(&_mapping->get_mapped()->get_parent<Inendi::PVSource>() ==
		       &_plotting->get_plotted()->get_parent<Inendi::PVSource>());
	} else {
		assert(has_mapping() || has_plotting());
	}
#endif
	if (has_mapping()) {
		_axes = &(_mapping->get_mapped()
		              ->get_parent<Inendi::PVSource>()
		              .current_view()
		              ->get_axes_combination()
		              .get_original_axes_list());
	} else {
		_axes = &(_plotting->get_plotted()
		              ->get_parent<Inendi::PVSource>()
		              .current_view()
		              ->get_axes_combination()
		              .get_original_axes_list());
	}

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

	QLabel* ret = new QLabel(text, NULL);
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
	connect(btns, SIGNAL(accepted()), this, SLOT(save_settings()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));
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
	name_layout->addWidget(new QLabel(tr("Name:"), NULL));
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

	int row = 1;
	PVCol col = 0;

	// We must get the official name of the
	QString name;
	if (has_mapping()) {
		name = QString::fromStdString(_mapping->get_name());
	} else {
		name = QString::fromStdString(_plotting->get_name());
	}
	_edit_name->setText(name);

	// Add widgets

	PVCol axis_id = 0;
	for (auto& axe : *_axes) {
		col = 0;
		_main_grid->addWidget(new QLabel(axe.get_name(), this), row, col++);
		if (has_mapping()) {
			PVWidgets::PVAxisTypeWidget* type_combo =
			    new PVWidgets::PVAxisTypeWidget(_mapping->get_type_for_col(axis_id), this);
			_main_grid->addWidget(type_combo, row, col++);
			connect(type_combo, SIGNAL(currentIndexChanged(const QString&)), this,
			        SLOT(type_changed(const QString&)));
			_main_grid->addWidget(new PVWidgets::PVMappingModeWidget(axis_id, *_mapping, this), row,
			                      col++);
		}
		if (has_plotting()) {
			_main_grid->addWidget(new PVWidgets::PVPlottingModeWidget(axis_id, *_plotting, this),
			                      row, col++);
		}
		axis_id++;
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
	PVCol axis_id = 0;
	Inendi::PVAxesCombination::list_axes_t::const_iterator it_axes;
	for (it_axes = _axes->begin(); it_axes != _axes->end(); it_axes++) {
		int col = 1;
		if (has_mapping()) {
			Inendi::PVMappingProperties& prop = _mapping->get_properties_for_col(axis_id);
			// Axis type
			PVWidgets::PVAxisTypeWidget* combo = dynamic_cast<PVWidgets::PVAxisTypeWidget*>(
			    _main_grid->itemAtPosition(row, col++)->widget());
			assert(combo);
			QString type = combo->get_sel_type();

			// Mapping mode
			PVWidgets::PVMappingModeWidget* map_combo =
			    dynamic_cast<PVWidgets::PVMappingModeWidget*>(
			        _main_grid->itemAtPosition(row, col++)->widget());
			assert(map_combo);
			QString mode = map_combo->get_mode();

			// TODO : We should cannot change type anymore.
			//			prop.set_type(type, mode);
			prop.set_mode(mode);
			prop.set_args(map_combo->get_cur_filter_params());
		}
		if (has_plotting()) {
			PVWidgets::PVPlottingModeWidget* combo = dynamic_cast<PVWidgets::PVPlottingModeWidget*>(
			    _main_grid->itemAtPosition(row, col++)->widget());
			assert(combo);
			QString mode = combo->get_mode();
			_plotting->get_properties_for_col(axis_id).set_mode(mode);
		}
		axis_id++;
		row++;
	}

	accept();
}

/******************************************************************************
 *
 * PVWidgets::PVMappingPlottingEditDialog::type_changed
 *
 *****************************************************************************/
void PVWidgets::PVMappingPlottingEditDialog::type_changed(const QString& type)
{
	PVLOG_DEBUG("PVWidgets::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);

	assert(has_mapping());
	PVWidgets::PVAxisTypeWidget* combo_org = dynamic_cast<PVWidgets::PVAxisTypeWidget*>(sender());
	assert(combo_org);
	int index = _main_grid->indexOf(combo_org);
	assert(index != -1);
	int row, col;
	int rspan, cspan;
	_main_grid->getItemPosition(index, &row, &col, &rspan, &cspan);
	// Mapping combo box is next to the type one
	PVWidgets::PVMappingModeWidget* combo_mapped = dynamic_cast<PVWidgets::PVMappingModeWidget*>(
	    _main_grid->itemAtPosition(row, col + 1)->widget());
	combo_mapped->clear();
	combo_mapped->populate_from_type(type);
	combo_mapped->select_default();
}
