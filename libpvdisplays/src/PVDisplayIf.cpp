#include <pvdisplays/PVDisplayIf.h>


// PVDisplayViewAxisIf implementation
QWidget* PVDisplays::PVDisplayViewAxisIf::get_unique_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent)
{
	QWidget* ret;
	map_widgets_t::const_iterator it = _widgets.find(Params(view, axis_comb));

	if (it == _widgets.end()) {
		ret = create_widget(view, axis_comb, parent);
		_widgets[Params(view, axis_comb)] = ret;
	}
	else {
		ret = it->second;
		assert(ret->parent() == parent);
	}

	return ret;
}

QAction* PVDisplays::PVDisplayViewAxisIf::action_bound_to_params(Picviz::PVView* view, PVCol axis_comb, QObject* parent) const
{
	QAction* action = new QAction(parent);

	Params p(view, axis_comb);
	QVariant var;
	var.setValue(p);
	action->setData(var);

	return action;
}

QWidget* PVDisplays::PVDisplayViewAxisIf::get_unique_widget_from_action(QAction const& action, QWidget* parent)
{
	Params p = get_params_from_action(action);
	return get_unique_widget(p.view, p.axis_comb, parent);
}

QWidget* PVDisplays::PVDisplayViewAxisIf::create_widget_from_action(QAction const& action, QWidget* parent) const
{
	Params p = get_params_from_action(action);
	return create_widget(p.view, p.axis_comb, parent);
}
