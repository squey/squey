#include <picviz/PVMappingFilter.h>
#include <picviz/PVView.h>

#include <PVMappingModeWidget.h>
#include <PVArgumentListWidget.h>

#include <QHBoxLayout>

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(QString const& type, QWidget* parent):
	QWidget(parent)
{
	init(false);
	populate_from_type(type);
}

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id, Picviz::PVMapping& mapping, bool params_btn, QWidget* parent):
	QWidget(parent)
{
	init(params_btn);
	populate_from_mapping(axis_id, mapping);
}

PVInspector::PVWidgetsHelpers::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id, Picviz::PVView& view, bool params_btn, QWidget* parent):
	QWidget(parent)
{
	init(params_btn);
	populate_from_mapping(axis_id, view.get_mapped_parent()->get_mapping());
}

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::init(bool params_btn)
{
	_combo = new PVComboBox(this);
	_props = NULL;
	
	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(_combo);
	if (params_btn) {
		_params_btn = new QPushButton(tr("Parameters..."));
		QSizePolicy sp(QSizePolicy::Maximum, QSizePolicy::Fixed);
		_params_btn->setSizePolicy(sp);
		layout->addWidget(_params_btn);

		connect(_params_btn, SIGNAL(clicked()), this, SLOT(change_params()));
	}
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

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::populate_from_mapping(PVCol axis_id, Picviz::PVMapping& mapping)
{
	Picviz::PVMappingProperties& props = mapping.get_properties_for_col(axis_id);
	_props = &props;
	populate_from_type(props.get_type());
	set_mode(props.get_mode());

	if (_props->get_args().size() == 0 && _params_btn) {
		_params_btn->setEnabled(false);
	}
}

QSize PVInspector::PVWidgetsHelpers::PVMappingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

void PVInspector::PVWidgetsHelpers::PVMappingModeWidget::change_params()
{
	if (!_props) {
		return;
	}

	// Get argument from the properties and modify them
	PVCore::PVArgumentList args = _props->get_args();
	if (args.size() == 0) {
		return;
	}
	bool ret = PVArgumentListWidget::modify_arguments_dlg(PVArgumentListWidget::create_mapping_plotting_widget_factory(), args, this);
	if (!ret) {
		return;
	}

	_props->set_args(args);
}
