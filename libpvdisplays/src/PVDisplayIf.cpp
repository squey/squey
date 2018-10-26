/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvdisplays/PVDisplayIf.h>

// PVDisplayViewAxisIf implementation
QWidget* PVDisplays::PVDisplayViewAxisIf::get_unique_widget(Inendi::PVView* view,
                                                            PVCombCol axis_comb,
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

QAction* PVDisplays::PVDisplayViewAxisIf::action_bound_to_params(Inendi::PVView* view,
                                                                 PVCombCol axis_comb,
                                                                 QObject* parent) const
{
	auto action = new QAction(parent);

	Params p(view, axis_comb);
	QVariant var;
	var.setValue(p);
	action->setData(var);

	return action;
}

// PVDisplayViewZoneIf implementation
QWidget* PVDisplays::PVDisplayViewZoneIf::get_unique_widget(Inendi::PVView* view,
                                                            PVCombCol axis_comb_first,
                                                            PVCombCol axis_comb_second,
                                                            QWidget* parent)
{
	QWidget* ret;
	map_widgets_t::const_iterator it =
	    _widgets.find(Params(view, axis_comb_first, axis_comb_second, true));

	if (it == _widgets.end()) {
		ret = create_widget(view, axis_comb_first, axis_comb_second, parent);
		_widgets[Params(view, axis_comb_first, axis_comb_second, true)] = ret;
	} else {
		ret = it->second;
		assert(ret->parent() == parent);
	}

	return ret;
}

QAction* PVDisplays::PVDisplayViewZoneIf::action_bound_to_params(Inendi::PVView* view,
                                                                 PVCombCol axis_comb_first,
                                                                 PVCombCol axis_comb_second,
                                                                 bool ask_for_box,
                                                                 QObject* parent) const
{
	auto action = new QAction(parent);

	Params p(view, axis_comb_first, axis_comb_second, ask_for_box);
	QVariant var;
	var.setValue(p);
	action->setData(var);

	return action;
}
