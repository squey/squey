#include <PVPlottingModeWidget.h>
#include <picviz/PVView.h>

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(QString const& type, QWidget* parent):
	PVComboBox(parent)
{
	populate_from_type(type);
}

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id, Picviz::PVPlotting const& plotting, QWidget* parent):
	PVComboBox(parent)
{
	populate_from_plotting(axis_id, plotting);
}

PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id, Picviz::PVView const& view, QWidget* parent):
	PVComboBox(parent)
{
	populate_from_plotting(axis_id, view.get_plotted_parent()->get_plotting());
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
			addItem(human_name, params[1]);
		}
	}
}

void PVInspector::PVWidgetsHelpers::PVPlottingModeWidget::populate_from_plotting(PVCol axis_id, Picviz::PVPlotting const& plotting)
{
	Picviz::PVPlottingProperties const& props = plotting.get_properties_for_col(axis_id);
	populate_from_type(props.get_type());
	set_mode(props.get_mode());
}
