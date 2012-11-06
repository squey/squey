#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <pvparallelview/PVDisplayViewAxisZoomed.h>

PVDisplays::PVDisplayViewAxisZoomed::PVDisplayViewAxisZoomed():
	PVDisplayViewAxisIf(PVDisplayIf::ShowInToolbar)
{
}

QWidget* PVDisplays::PVDisplayViewAxisZoomed::create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_zoomed_view(axis_comb, parent);

	return widget;
}
