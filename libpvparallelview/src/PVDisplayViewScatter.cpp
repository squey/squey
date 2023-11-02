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



#include <pvdisplays/PVDisplaysContainer.h>
#include <pvkernel/widgets/PVFilterableMenu.h>

#include <pvparallelview/PVDisplayViewScatter.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVScatterView.h>

#include <pvkernel/widgets/PVModdedIcon.h>

PVDisplays::PVDisplayViewScatter::PVDisplayViewScatter()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu | PVDisplayIf::HasHelpPage,
                      "Scatter view",
                      PVModdedIcon("scatter"),
                      "New scatter view with axis...")
{
}

QWidget* PVDisplays::PVDisplayViewScatter::create_widget(Squey::PVView* view,
                                                         QWidget* parent,
                                                         Params const& params) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_scatter_view(col_param(view, params, 0),
	                                                col_param(view, params, 1), parent);

	return widget;
}

void PVDisplays::PVDisplayViewScatter::add_to_axis_menu(QMenu& menu,
                                                        PVCol axis,
                                                        PVCombCol axis_comb,
                                                        Squey::PVView* view,
                                                        PVDisplays::PVDisplaysContainer* container)
{
	if (axis_comb == PVCombCol()) {
		auto act = new QAction(toolbar_icon(), axis_menu_name());
		act->connect(act, &QAction::triggered, [container, this, view, axis]() {
			container->create_view_widget(*this, view, {axis, PVCol()});
		});
		menu.addAction(act);
		return;
	}

	auto* axes_menu =
	    new PVWidgets::PVFilterableMenu(axis_menu_name(), &menu);
	QList<QAction*> actions;
	QAction* next_axis = nullptr;

	const QStringList& axes_names = view->get_axes_names_list();

	for (PVCombCol i(0); i < view->get_axes_combination().get_axes_count(); i++) {
		if (i != axis_comb) {
			auto create_action = [&]() {
				auto* act = new QAction(axes_names[i]);
				act->connect(act, &QAction::triggered, [container, this, view, axis_comb, i]() {
					container->create_view_widget(*this, view, {axis_comb, PVCombCol(i)});
				});
				return act;
			};

			actions << create_action();

			if (i == (axis_comb + 1)) {
				next_axis = create_action();
			}
		}
	}

	axes_menu->setIcon(toolbar_icon());
	axes_menu->addAction(next_axis); // Shortcut for next axis
	axes_menu->addSeparator();
	axes_menu->addActions(actions);
	menu.addMenu(axes_menu);
}
