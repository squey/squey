#include <picviz/PVView.h>

#include <PVPlottingModeWidget.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

#include <QHBoxLayout>

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(QString const& type, QWidget* parent):
	QWidget(parent)
{
	init(false);
	populate_from_type(type);
}

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id, Picviz::PVPlotting& plotting, bool params_btn, QWidget* parent):
	QWidget(parent)
{
	init(params_btn);
	populate_from_plotting(axis_id, plotting);
}

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id, Picviz::PVView& view, bool params_btn, QWidget* parent):
	QWidget(parent)
{
	init(params_btn);
	populate_from_plotting(axis_id, *view.get_parent<Picviz::PVPlotted>()->get_plotting());
}

QSize PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

void PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::init(bool params_btn)
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
	else {
		_params_btn = NULL;
	}
	setLayout(layout);
	
	setFocusPolicy(Qt::StrongFocus);
}

void PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Picviz::PVPlottingFilter)::list_classes const& map_filters = LIB_CLASS(Picviz::PVPlottingFilter)::get().get_list();
	LIB_CLASS(Picviz::PVPlottingFilter)::list_classes::const_iterator it;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		Picviz::PVPlottingFilter::p_type filter = *it;
		QString const& name = it.key();
		QString human_name = (*it)->get_human_name();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			_combo->addItem(human_name, params[1]);
		}
	}
}

void PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::populate_from_plotting(PVCol axis_id, Picviz::PVPlotting& plotting)
{
	Picviz::PVPlottingProperties& props = plotting.get_properties_for_col(axis_id);
	_props = &props;
	populate_from_type(props.get_type());
	set_mode(props.get_mode());

	if (_props->get_args().size() == 0 && _params_btn) {
		_params_btn->setEnabled(false);
	}
}

void PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::change_params()
{
	if (!_props) {
		return;
	}

	// Get argument from the properties and modify them
	PVCore::PVArgumentList args = _props->get_args();
	if (args.size() == 0) {
		return;
	}
	bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), args, this);
	if (!ret) {
		return;
	}

	_props->set_args(args);
}
