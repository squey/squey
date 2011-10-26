#include <PVMappingPlottingEditDialog.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVSource.h>

#include <QLabel>
#include <QDialogButtonBox>

PVInspector::PVMappingPlottingEditDialog::PVMappingPlottingEditDialog(Picviz::PVMapping* mapping, Picviz::PVPlotting* plotting, QWidget* parent):
	QDialog(parent),
	_mapping(mapping),
	_plotting(plotting)
{
#ifndef NDEBUG
	if (has_mapping() && has_plotting()) {
		assert(_mapping->get_source_parent() == _plotting->get_source_parent());
	}
	else {
		assert(has_mapping() || has_plotting());
	}
#endif
	if (has_mapping()) {
		_format = &(_mapping->get_source_parent()->get_format());
	}
	else {
		_format = &(_plotting->get_source_parent()->get_format());
	}

	init_layout();
	load_settings();
	finish_layout();
}

PVInspector::PVMappingPlottingEditDialog::~PVMappingPlottingEditDialog()
{
}

void PVInspector::PVMappingPlottingEditDialog::init_layout()
{
	_main_layout = new QVBoxLayout();
	_main_grid = new QGridLayout();
	int row = 0;
	int col = 0;

	// Init titles
	_main_grid->addWidget(new QLabel(tr("Axes"), this), row, col);
	col++;
	if (has_mapping()) {
		_main_grid->addWidget(new QLabel(tr("Type"), this), row, col);
		col++;
		_main_grid->addWidget(new QLabel(tr("Mapping"), this), row, col);
		col++;
	}
	if (has_plotting()) {
		_main_grid->addWidget(new QLabel(tr("Plotting"), this), row, col);
		col++;
	}
	_main_layout->addLayout(_main_grid);
	setLayout(_main_layout);
}

void PVInspector::PVMappingPlottingEditDialog::load_settings()
{
	int row = 1;
	PVCol col = 0;
	// Add widgets

	PVRush::list_axes_t const& axes = _format->get_axes();
	PVRush::list_axes_t::const_iterator it_axes;
	PVCol axis_id = 0;
	for (it_axes = axes.begin(); it_axes != axes.end(); it_axes++) {
		col = 0;
		_main_grid->addWidget(new QLabel(it_axes->get_name(), this), row, col++);
		if (has_mapping()) {
			Picviz::PVMappingProperties const& prop = _mapping->get_properties_for_col(axis_id);
			QComboBox* type_combo = init_combo(get_list_types(), prop.get_type());
			_main_grid->addWidget(type_combo, row, col++);
			connect(type_combo, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(type_changed(const QString&)));
			_main_grid->addWidget(init_combo(get_list_mapping(prop.get_type()), prop.get_mode()), row, col++);
		}
		if (has_plotting()) {
			Picviz::PVPlottingProperties const& prop = _plotting->get_properties_for_col(axis_id);
			_main_grid->addWidget(init_combo(get_list_plotting(prop.get_type()), prop.get_mode()), row, col++);
		}
		axis_id++;
		row++;
	}
}

void PVInspector::PVMappingPlottingEditDialog::finish_layout()
{
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	_main_layout->addWidget(btns);
}

QComboBox* PVInspector::PVMappingPlottingEditDialog::init_combo(QStringList const& list, QString const& sel)
{
	QComboBox* ret = new QComboBox();
	ret->addItems(list);
	assert(list.contains(sel));
	ret->setCurrentIndex(list.indexOf(sel));
	return ret;
}

QStringList PVInspector::PVMappingPlottingEditDialog::get_list_types()
{
	return Picviz::PVMappingFilter::list_types();
}

QStringList PVInspector::PVMappingPlottingEditDialog::get_list_mapping(QString const& type)
{
	return Picviz::PVMappingFilter::list_modes(type);
}

QStringList PVInspector::PVMappingPlottingEditDialog::get_list_plotting(QString const& type)
{
	return Picviz::PVPlottingFilter::list_modes(type);
}

void PVInspector::PVMappingPlottingEditDialog::type_changed(const QString& type)
{
	assert(has_mapping());
	QComboBox* combo_org = dynamic_cast<QComboBox*>(sender());
	assert(combo_org);
	int index = _main_grid->indexOf(combo_org);
	assert(index != -1);
	int row,col;
	int rspan,cspan;
	_main_grid->getItemPosition(index, &row, &col, &rspan, &cspan);
	// Mapping combo box is next to the type one
	QComboBox* combo_mapped = dynamic_cast<QComboBox*>(_main_grid->itemAtPosition(row, col+1)->widget());
	combo_mapped->clear();
	combo_mapped->addItems(get_list_mapping(type));
}
