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
    : PVDisplayViewDataIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                          "Zoomed parallel view")
{
}

QWidget* PVDisplays::PVDisplayViewZoomedParallel::create_widget(Inendi::PVView* view,
                                                                Params const& data,
                                                                QWidget* parent) const
{
	auto axis_comb = data.at(0);
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_zoomed_view(axis_comb, parent);

	return widget;
}

QIcon PVDisplays::PVDisplayViewZoomedParallel::toolbar_icon() const
{
	return QIcon(":/view-parallel-zoomed");
}

// FIXME : Hidden argument reflect bad design properties, inheritance should certainly be improved.
QString PVDisplays::PVDisplayViewZoomedParallel::widget_title(Inendi::PVView* view,
                                                              Params const&) const
{
	return "Zoomed view [" + QString::fromStdString(view->get_name()) + "]";
}

QString PVDisplays::PVDisplayViewZoomedParallel::axis_menu_name(Inendi::PVView*,
                                                                Params const&) const
{
	return QString("New zoomed parallel view");
}
