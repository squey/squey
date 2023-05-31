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

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvparallelview/PVDisplayViewZoomedParallel.h>

PVDisplays::PVDisplayViewZoomedParallel::PVDisplayViewZoomedParallel()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                      "Zoomed parallel view",
                      QIcon(":/view-parallel-zoomed"),
                      "New zoomed parallel view")
{
}

QWidget* PVDisplays::PVDisplayViewZoomedParallel::create_widget(Squey::PVView* view,
                                                                QWidget* parent,
                                                                Params const& data) const
{
	auto axis_comb = data.size() > 0 ? std::any_cast<PVCombCol>(data.at(0)) : PVCombCol();
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_zoomed_view(axis_comb, parent);

	return widget;
}

void PVDisplays::PVDisplayViewZoomedParallel::add_to_axis_menu(
    QMenu& menu,
    PVCol,
    PVCombCol axis_comb,
    Squey::PVView* view,
    PVDisplays::PVDisplaysContainer* container)
{
	if (axis_comb == PVCombCol()) {
		return;
	}
	auto* act = new QAction(toolbar_icon(), axis_menu_name());
	act->connect(act, &QAction::triggered, [this, view, axis_comb, container]() {
		container->create_view_widget(*this, view, {axis_comb});
	});
	menu.addAction(act);
}
