#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvparallelview/PVDisplayViewFullParallel.h>

PVDisplays::PVDisplayViewFullParallel::PVDisplayViewFullParallel():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar)
{
}

QWidget* PVDisplays::PVDisplayViewFullParallel::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_view(parent);

	return widget;
}
