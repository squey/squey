/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVHitCountView.h>

#include <pvparallelview/PVDisplayViewHitCount.h>

PVDisplays::PVDisplayViewHitCount::PVDisplayViewHitCount()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                      "Hit count view",
                      QIcon(":/view-hit-count"),
                      "New hit count view")
{
}

QWidget* PVDisplays::PVDisplayViewHitCount::create_widget(Inendi::PVView* view,
                                                          QWidget* parent,
                                                          Params const& params) const
{
	PVParallelView::PVLibView* lib_view = PVParallelView::common::get_lib_view(*view);
	QWidget* widget = lib_view->create_hit_count_view(col_param(view, params, 0), parent);

	return widget;
}
