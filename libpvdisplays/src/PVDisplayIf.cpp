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
#include <pvkernel/widgets/PVModdedIcon.h>
#include <squey/PVView.h>

#include <QApplication>
#include <QClipboard>

namespace PVDisplays
{

QString PVDisplayViewIf::default_window_title(Squey::PVView& view) const
{
	return QString("%1 [%2]").arg(
		tooltip_str(),
		QString::fromStdString(view.get_name()));
}

void PVDisplayViewIf::add_to_axis_menu(QMenu& menu,
                                       PVCol axis,
                                       PVCombCol /*axis_comb*/,
                                       Squey::PVView* view,
                                       PVDisplaysContainer* container)
{
	auto* act = new QAction(toolbar_icon(), axis_menu_name());
	act->connect(act, &QAction::triggered, [this, view, axis, container]() {
		container->create_view_widget(*this, view, {axis});
	});
	menu.addAction(act);
}

void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Squey::PVView* view,
                                 PVCol axis,
                                 PVCombCol axis_comb)
{
	auto action_col_copy = new QAction(QObject::tr("Copy axis name to clipboard"), &menu);
	action_col_copy->setIcon(PVModdedIcon("copy"));
	QObject::connect(action_col_copy, &QAction::triggered, [view, axis_comb]{
		QApplication::clipboard()->setText(view->get_axis_name(axis_comb));
	});
    menu.addAction(action_col_copy);
	menu.addSeparator();

	visit_displays_by_if<PVDisplayViewIf>(
	    [&](PVDisplayViewIf& interface) {
		    interface.add_to_axis_menu(menu, axis, axis_comb, view, container);
		},
	    PVDisplayIf::ShowInCtxtMenu);
}

void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Squey::PVView* view,
                                 PVCombCol axis_comb)
{
	assert(axis_comb != PVCombCol());
	add_displays_view_axis_menu(menu, container, view,
	                            view->get_axes_combination().get_nraw_axis(axis_comb), axis_comb);
}

PVCol col_param(Squey::PVView* view, std::vector<std::any> const& params, size_t index)
{
	if (index >= params.size()) {
		return {};
	}
	if (auto* comb_col = std::any_cast<PVCombCol>(&params[index])) {
		return view->get_axes_combination().get_nraw_axis(*comb_col);
	}
	return std::any_cast<PVCol>(params[index]);
};
}
