
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvparallelview/PVDisplayViewZoomedParallel.h>

PVDisplays::PVDisplayViewZoomedParallel::PVDisplayViewZoomedParallel():
	PVDisplayViewAxisIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu, "Zoomed parallel view")
{
}

QWidget* PVDisplays::PVDisplayViewZoomedParallel::create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_zoomed_view(axis_comb, parent);

	return widget;
}

QIcon PVDisplays::PVDisplayViewZoomedParallel::toolbar_icon() const
{
	return QIcon(":/view-parallel-zoomed");
}

QString PVDisplays::PVDisplayViewZoomedParallel::widget_title(Picviz::PVView* view, PVCol axis_comb) const
{
	return QString("Zoomed view [" + view->get_name() + " on axis " + view->get_axis_name(axis_comb) + "]");
}

QString PVDisplays::PVDisplayViewZoomedParallel::axis_menu_name(Picviz::PVView const* /*view*/, PVCol /*axis_comb*/) const
{
	return QString("New zoomed parallel view");
}
