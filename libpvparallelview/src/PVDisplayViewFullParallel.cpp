#include <picviz/PVView.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvparallelview/PVDisplayViewFullParallel.h>

PVDisplays::PVDisplayViewFullParallel::PVDisplayViewFullParallel():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget | PVDisplayIf::DefaultPresenceInSourceWorkspace, "Full parallel view", Qt::TopDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewFullParallel::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_view(parent);

	return widget;
}

QIcon PVDisplays::PVDisplayViewFullParallel::toolbar_icon() const
{
	return QIcon(":/view_display_parallel");
}

QString PVDisplays::PVDisplayViewFullParallel::widget_title(Picviz::PVView* view) const
{
	return QString("Parallel view [" + view->get_name() + "]");
}
