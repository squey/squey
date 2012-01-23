#include <PVMappingModeWidget.h>
#include <picviz/PVMappingFilter.h>
#include <picviz/PVView.h>

#include <QHBoxLayout>

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(QString const& type, QWidget* parent):
	QWidget(parent)
{
	init();
	_params_btn->setVisible(false);
	populate_from_type(type);
}

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id, Picviz::PVMapping const& mapping, QWidget* parent):
	QWidget(parent)
{
	init();
	populate_from_mapping(axis_id, mapping);
}

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id, Picviz::PVView const& view, QWidget* parent):
	QWidget(parent)
{
	init();
	populate_from_mapping(axis_id, view.get_mapped_parent()->get_mapping());
}

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::init()
{
	_combo = new PVComboBox(this);
	_params_btn = new QPushButton(tr("Parameters..."));
	
	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(_combo);
	layout->addWidget(_params_btn);
	setLayout(layout);
	
	setFocusPolicy(Qt::StrongFocus);
}

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes const& map_filters = LIB_CLASS(Picviz::PVMappingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVMappingFilter)::list_classes::const_iterator it;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		Picviz::PVMappingFilter::p_type filter = *it;
		QString const& name = it.key();
		QString human_name = (*it)->get_human_name();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			_combo->addItem(human_name, params[1]);
		}
	}
}

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::populate_from_mapping(PVCol axis_id, Picviz::PVMapping const& mapping)
{
	Picviz::PVMappingProperties const& props = mapping.get_properties_for_col(axis_id);
	populate_from_type(props.get_type());
	set_mode(props.get_mode());
}

QSize PVInspector::PVWidgetsHelpers::PVMappingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}
