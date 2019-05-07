/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>

#include <pvparallelview/PVDisplayViewTimeseries.h>
#include <pvparallelview/PVSeriesViewWidget.h>

PVDisplays::PVDisplayViewTimeseries::PVDisplayViewTimeseries()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                      "Series view",
                      QIcon(":/view-series"),
                      "New series view")
{
}

QWidget* PVDisplays::PVDisplayViewTimeseries::create_widget(Inendi::PVView* view,
                                                            QWidget* parent,
                                                            Params const& params) const
{
	return new PVParallelView::PVSeriesViewWidget(view, col_param(view, params, 0), parent);
}

void PVDisplays::PVDisplayViewTimeseries::add_to_axis_menu(
    QMenu& menu,
    PVCol axis,
    PVCombCol axis_comb,
    Inendi::PVView* view,
    PVDisplays::PVDisplaysContainer* container)
{
	if (view->get_axes_combination().get_axis(axis).get_type().left(4) == "time" or
	    view->get_axes_combination().get_axis(axis).get_type().left(7) == "number_") {
		PVDisplayViewIf::add_to_axis_menu(menu, axis, axis_comb, view, container);
	}
}
