/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVProgressBox.h>

#include <inendi/PVView.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvparallelview/PVDisplayViewFullParallel.h>

PVDisplays::PVDisplayViewFullParallel::PVDisplayViewFullParallel()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                          PVDisplayIf::DefaultPresenceInSourceWorkspace,
                      "Full parallel view",
                      Qt::TopDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewFullParallel::create_widget(Inendi::PVView* view,
                                                              QWidget* parent) const
{
	PVParallelView::PVLibView* lib_view;

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(false);
		    lib_view = PVParallelView::common::get_lib_view(*view);
		},
	    "Initializing full parallel view", parent);

	return lib_view->create_view(parent);
	;
}

QIcon PVDisplays::PVDisplayViewFullParallel::toolbar_icon() const
{
	return QIcon(":/view-parallel-full");
}

QString PVDisplays::PVDisplayViewFullParallel::widget_title(Inendi::PVView* view) const
{
	return "Parallel view [" + QString::fromStdString(view->get_name()) + "]";
}
