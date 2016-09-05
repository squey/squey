/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvdisplays/PVDisplayIf.h>

// PVDisplayViewAxisIf implementation
QWidget* PVDisplays::__impl::PVDisplayViewAxisIf::get_unique_widget(Inendi::PVView* view,
                                                                    Inendi::PVCombCol axis_comb,
                                                                    QWidget* parent)
{
	QWidget* ret;
	map_widgets_t::const_iterator it = _widgets.find(Params(view, axis_comb));

	if (it == _widgets.end()) {
		ret = create_widget(view, axis_comb, parent);
		_widgets[Params(view, axis_comb)] = ret;
	} else {
		ret = it->second;
		assert(ret->parent() == parent);
	}

	return ret;
}

QAction* PVDisplays::__impl::PVDisplayViewAxisIf::action_bound_to_params(
    Inendi::PVView* view, Inendi::PVCombCol axis_comb, QObject* parent) const
{
	QAction* action = new QAction(parent);

	Params p(view, axis_comb);
	QVariant var;
	var.setValue(p);
	action->setData(var);

	return action;
}
