#include <PVMappingPlottingEditDialog.h>
#include <picviz/PVMapping.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVPlottingFilter.h>
#include <picviz/PVPlotting.h>
#include <picviz/PVSource.h>

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QScrollArea>

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

	setWindowTitle(tr("Edit properties..."));

	init_layout();
	load_settings();
	finish_layout();
}

PVInspector::PVMappingPlottingEditDialog::~PVMappingPlottingEditDialog()
{
}

QLabel* PVInspector::PVMappingPlottingEditDialog::create_label(QString const& text, Qt::Alignment align)
{
	QLabel* ret = new QLabel(text, NULL);
	ret->setAlignment(align);
	QFont font(ret->font());
	font.setBold(true);
	ret->setFont(font);
	return ret;
}

void PVInspector::PVMappingPlottingEditDialog::init_layout()
{
	_main_layout = new QVBoxLayout();
	_main_layout->setSpacing(29);

	QHBoxLayout* name_layout = new QHBoxLayout();
	name_layout->addWidget(new QLabel(tr("Name:"), NULL));
	_edit_name = new QLineEdit();
	name_layout->addWidget(_edit_name);
	_main_layout->addLayout(name_layout);

	QScrollArea* scroll_area = new QScrollArea();
	scroll_area->setWidgetResizable(true);
	_main_grid = new QGridLayout(scroll_area);
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

	QVBoxLayout* scroll_layout = new QVBoxLayout();
	scroll_layout->addWidget(scroll_area);

	QGroupBox* box = new QGroupBox(tr("Parameters"));
	box->setLayout(scroll_layout);
	_main_layout->addWidget(scroll_area);

	setLayout(_main_layout);
}

void PVInspector::PVMappingPlottingEditDialog::load_settings()
{
	int row = 1;
	PVCol col = 0;
	// Name
	QString name;
	if (has_mapping()) {
		name = _mapping->get_name();
	}
	else {
		name = _plotting->get_name();
	}
	_edit_name->setText(name);

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

void PVInspector::PVMappingPlottingEditDialog::save_settings()
{
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
	PVCol axis_id;
	for (axis_id = 0; axis_id < _format->get_axes().size(); axis_id++) {
		int col = 1;
		if (has_mapping()) {
			Picviz::PVMappingProperties& prop = _mapping->get_properties_for_col(axis_id);
			// Axis type
			QComboBox* combo = dynamic_cast<QComboBox*>(_main_grid->itemAtPosition(row, col++)->widget());
			QString type = combo->currentText();

			// Mapping mode
			combo = dynamic_cast<QComboBox*>(_main_grid->itemAtPosition(row, col++)->widget());
			QString mode = combo->currentText();

			prop.set_type(type, mode);
		}
		if (has_plotting()) {
			QComboBox* combo = dynamic_cast<QComboBox*>(_main_grid->itemAtPosition(row, col++)->widget());
			QString mode = combo->currentText();
			_plotting->get_properties_for_col(axis_id).set_mode(mode);
		}
		row++;
	}

	accept();
}

void PVInspector::PVMappingPlottingEditDialog::finish_layout()
{
	QDialogButtonBox* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	connect(btns, SIGNAL(accepted()), this, SLOT(save_settings()));
	connect(btns, SIGNAL(rejected()), this, SLOT(reject()));
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
