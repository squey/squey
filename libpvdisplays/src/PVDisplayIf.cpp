//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvdisplays/PVDisplayIf.h>

#include <inendi/PVView.h>

namespace PVDisplays
{

QString PVDisplayViewIf::widget_title(Inendi::PVView* view) const
{
	return tooltip_str() + " [" + QString::fromStdString(view->get_name()) + "]";
}

void PVDisplayViewIf::add_to_axis_menu(QMenu& menu,
                                       PVCol axis,
                                       PVCombCol /*axis_comb*/,
                                       Inendi::PVView* view,
                                       PVDisplaysContainer* container)
{
	QAction* act = new QAction(toolbar_icon(), axis_menu_name());
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
