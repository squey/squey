/**
 * \file PVMappingPlottingEditDialog.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <PVMappingPlottingEditDialog.h>
#include <PVAxisTypeWidget.h>
#include <PVMappingModeWidget.h>
#include <PVPlottingModeWidget.h>

#include <pvkernel/core/general.h>

#include <picviz/PVMapping.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QScrollArea>



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog
 *
 *****************************************************************************/
PVInspector::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog(Picviz::PVMapping* mapping, Picviz::PVPlotting* plotting, QWidget* parent):
	QDialog(parent),
	_mapping(mapping),
	_plotting(plotting)
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
#ifndef NDEBUG
	if (has_mapping() && has_plotting()) {
		assert(_mapping->get_source_parent() == _plotting->get_source_parent());
	}
	else {
		assert(has_mapping() || has_plotting());
	}
#endif
	if (has_mapping()) {
		_axes = &(_mapping->get_source_parent()->current_view()->axes_combination.get_original_axes_list());
	}
	else {
		_axes = &(_plotting->get_source_parent()->current_view()->axes_combination.get_original_axes_list());
	}

	// We set the nale of the Dialog Window
	setWindowTitle(tr("Edit properties..."));

	// We generate and populate the Layout
	init_layout();
	// We load the current settings
	load_settings();
	// We can set the layout in the Widget
	finish_layout();
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::~PVMappingPlottingEditDialog
 *
 *****************************************************************************/
PVInspector::PVMappingPlottingEditDialog::~PVMappingPlottingEditDialog()
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::create_label
 *
 *****************************************************************************/
QLabel* PVInspector::PVMappingPlottingEditDialog::create_label(QString const& text, Qt::Alignment align)
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	QLabel* ret = new QLabel(text, NULL);
	ret->setAlignment(align);
	QFont font(ret->font());
	font.setBold(true);
	ret->setFont(font);
	return ret;
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::finish_layout
 *
 *****************************************************************************/
void PVInspector::PVMappingPlottingEditDialog::finish_layout()
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(btns, SIGNAL(accepted()), this, SLOT(save_settings()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));
	_main_layout->addWidget(btns);
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::init_layout
 *
 *****************************************************************************/
void PVInspector::PVMappingPlottingEditDialog::init_layout()
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	// We need a QVBoxLayout for that Widget
	_main_layout = new QVBoxLayout();
	_main_layout->setSpacing(29);

	
	QHBoxLayout* name_layout = new QHBoxLayout();
	name_layout->addWidget(new QLabel(tr("Name:"), NULL));
	_edit_name = new QLineEdit();
	name_layout->addWidget(_edit_name);
	_main_layout->addLayout(name_layout);

	QWidget* grid_widget = new QWidget();
	_main_grid = new QGridLayout(grid_widget);
	_main_grid->setHorizontalSpacing(20);
	_main_grid->setVerticalSpacing(10);
	int row = 0;
	int col = 0;

	// Init titles
	_main_grid->addWidget(create_label(tr("Axis"), Qt::AlignLeft), row, col);
	col++;
	if (has_mapping()) {
		_main_grid->addWidget(create_label(tr("Type")), row, col);
		col++;
		_main_grid->addWidget(create_label(tr("Mapping")), row, col);
		col++;
	}
	if (has_plotting()) {
		_main_grid->addWidget(create_label(tr("Plotting")), row, col);
		col++;
	}
	QScrollArea* scroll_area = new QScrollArea();
	scroll_area->setWidget(grid_widget);
	scroll_area->setWidgetResizable(true);

	QVBoxLayout* scroll_layout = new QVBoxLayout();
	scroll_layout->addWidget(scroll_area);

	QGroupBox* box = new QGroupBox(tr("Parameters"));
	box->setLayout(scroll_layout);
	_main_layout->addWidget(box);

	setLayout(_main_layout);
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::load_settings
 *
 *****************************************************************************/
void PVInspector::PVMappingPlottingEditDialog::load_settings()
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	int row = 1;
	PVCol col = 0;
	
	// We must get the official name of the 
	QString name;
	if (has_mapping()) {
		name = _mapping->get_name();
	}
	else {
		name = _plotting->get_name();
	}
	_edit_name->setText(name);

	// Add widgets

	Picviz::PVAxesCombination::list_axes_t::const_iterator it_axes;
	PVCol axis_id = 0;
	for (it_axes = _axes->begin(); it_axes != _axes->end(); it_axes++) {
		col = 0;
		_main_grid->addWidget(new QLabel(it_axes->get_name(), this), row, col++);
		if (has_mapping()) {
			PVWidgetsHelpers::PVAxisTypeWidget* type_combo = new PVWidgetsHelpers::PVAxisTypeWidget(this);
			type_combo->sel_type(_mapping->get_type_for_col(axis_id));
			_main_grid->addWidget(type_combo, row, col++);
			connect(type_combo, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(type_changed(const QString&)));
			_main_grid->addWidget(new PVWidgetsHelpers::PVMappingModeWidget(axis_id, *_mapping, this), row, col++);
		}
		if (has_plotting()) {
			_main_grid->addWidget(new PVWidgetsHelpers::PVPlottingModeWidget(axis_id, *_plotting, this), row, col++);
		}
		axis_id++;
		row++;
	}
}



/******************************************************************************
 *
 * PVInspector::PVMappingPlottingEditDialog::save_settings
 *
 *****************************************************************************/
void PVInspector::PVMappingPlottingEditDialog::save_settings()
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	QString name = _edit_name->text();
	if (name.isEmpty()) {
		_edit_name->setFocus(Qt::MouseFocusReason);
		return;
	}

	// If we're editing both at the same time, give the same name !
	if (has_mapping()) {
		_mapping->set_name(name);
	}
	if (has_plotting()) {
		_plotting->set_name(name);
	}

	int row = 1;
	PVCol axis_id = 0;
	Picviz::PVAxesCombination::list_axes_t::const_iterator it_axes;
	for (it_axes = _axes->begin(); it_axes != _axes->end(); it_axes++) {
		int col = 1;
		if (has_mapping()) {
			Picviz::PVMappingProperties& prop = _mapping->get_properties_for_col(axis_id);
			// Axis type
			PVWidgetsHelpers::PVAxisTypeWidget* combo = dynamic_cast<PVWidgetsHelpers::PVAxisTypeWidget*>(_main_grid->itemAtPosition(row, col++)->widget());
			assert(combo);
			QString type = combo->get_sel_type();

			// Mapping mode
			PVWidgetsHelpers::PVMappingModeWidget* map_combo = dynamic_cast<PVWidgetsHelpers::PVMappingModeWidget*>(_main_grid->itemAtPosition(row, col++)->widget());
			assert(map_combo);
			QString mode = map_combo->get_mode();

			prop.set_type(type, mode);
		}
		if (has_plotting()) {
			PVWidgetsHelpers::PVPlottingModeWidget* combo = dynamic_cast<PVWidgetsHelpers::PVPlottingModeWidget*>(_main_grid->itemAtPosition(row, col++)->widget());
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
 * PVInspector::PVMappingPlottingEditDialog::type_changed
 *
 *****************************************************************************/
void PVInspector::PVMappingPlottingEditDialog::type_changed(const QString& type)
{
	PVLOG_DEBUG("PVInspector::PVMappingPlottingEditDialog::%s\n", __FUNCTION__);
	
	assert(has_mapping());
	PVWidgetsHelpers::PVAxisTypeWidget* combo_org = dynamic_cast<PVWidgetsHelpers::PVAxisTypeWidget*>(sender());
	assert(combo_org);
	int index = _main_grid->indexOf(combo_org);
	assert(index != -1);
	int row,col;
	int rspan,cspan;
	_main_grid->getItemPosition(index, &row, &col, &rspan, &cspan);
	// Mapping combo box is next to the type one
	PVWidgetsHelpers::PVMappingModeWidget* combo_mapped = dynamic_cast<PVWidgetsHelpers::PVMappingModeWidget*>(_main_grid->itemAtPosition(row, col+1)->widget());
	combo_mapped->clear();
	combo_mapped->populate_from_type(type);
	combo_mapped->select_default();
}
