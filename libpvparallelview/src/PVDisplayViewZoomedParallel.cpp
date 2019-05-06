/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvparallelview/PVDisplayViewZoomedParallel.h>

PVDisplays::PVDisplayViewZoomedParallel::PVDisplayViewZoomedParallel()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                      "Zoomed parallel view")
{
}

QWidget* PVDisplays::PVDisplayViewZoomedParallel::create_widget(Inendi::PVView* view,
                                                                QWidget* parent,
                                                                Params const& data) const
{
	auto axis_comb = data.size() > 0 ? std::any_cast<PVCombCol>(data.at(0)) : PVCombCol();
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_zoomed_view(axis_comb, parent);

	return widget;
}

QIcon PVDisplays::PVDisplayViewZoomedParallel::toolbar_icon() const
{
	return QIcon(":/view-parallel-zoomed");
}

QString PVDisplays::PVDisplayViewZoomedParallel::widget_title(Inendi::PVView* view) const
{
	return "Zoomed view [" + QString::fromStdString(view->get_name()) + "]";
}

QString PVDisplays::PVDisplayViewZoomedParallel::axis_menu_name(Inendi::PVView*) const
{
	return QString("New zoomed parallel view");
}

void PVDisplays::PVDisplayViewZoomedParallel::add_to_axis_menu(
    QMenu& menu,
    PVCol,
    PVCombCol axis_comb,
    Inendi::PVView* view,
    PVDisplays::PVDisplaysContainer* container)
{
	if (axis_comb == PVCombCol()) {
		return;
	}
	QAction* act = new QAction(toolbar_icon(), axis_menu_name(view));
	act->connect(act, &QAction::triggered, [this, view, axis_comb, container]() {
		container->create_view_widget(*this, view, {axis_comb});
	});
	menu.addAction(act);
}
