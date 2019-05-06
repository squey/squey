/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvdisplays/PVDisplayIf.h>

#include <inendi/PVView.h>

namespace PVDisplays
{

void PVDisplayViewIf::add_to_axis_menu(QMenu& menu,
                                       PVCol axis,
                                       PVCombCol /*axis_comb*/,
                                       Inendi::PVView* view,
                                       PVDisplaysContainer* container)
{
	QAction* act = new QAction(toolbar_icon(), axis_menu_name(view));
	act->connect(act, &QAction::triggered, [this, view, axis, container]() {
		container->create_view_widget(*this, view, {axis});
	});
	menu.addAction(act);
}

void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Inendi::PVView* view,
                                 PVCol axis,
                                 PVCombCol axis_comb)
{
	visit_displays_by_if<PVDisplayViewIf>(
	    [&](PVDisplayViewIf& interface) {
		    interface.add_to_axis_menu(menu, axis, axis_comb, view, container);
		},
	    PVDisplayIf::ShowInCtxtMenu);
}

void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Inendi::PVView* view,
                                 PVCombCol axis_comb)
{
	assert(axis_comb != PVCombCol());
	add_displays_view_axis_menu(menu, container, view,
	                            view->get_axes_combination().get_nraw_axis(axis_comb), axis_comb);
}

PVCol col_param(Inendi::PVView* view, std::vector<std::any> const& params, size_t index)
{
	if (index >= params.size()) {
		return PVCol();
	}
	if (auto* comb_col = std::any_cast<PVCombCol>(&params[index])) {
		return view->get_axes_combination().get_nraw_axis(*comb_col);
	}
	return std::any_cast<PVCol>(params[index]);
};
}