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
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu, "Series view")
{
}

QWidget* PVDisplays::PVDisplayViewTimeseries::create_widget(Inendi::PVView* view,
                                                            QWidget* parent,
                                                            Params const& params) const
{
	return new PVParallelView::PVSeriesViewWidget(view, col_param(view, params, 0), parent);
}

QIcon PVDisplays::PVDisplayViewTimeseries::toolbar_icon() const
{
	return QIcon(":/view-series");
}

QString PVDisplays::PVDisplayViewTimeseries::widget_title(Inendi::PVView* view) const
{
	return "Series view [" + QString::fromStdString(view->get_name()) + /*" on axis '" +
	       view->get_axis_name(axis_comb) + */ "']";
}

QString PVDisplays::PVDisplayViewTimeseries::axis_menu_name(Inendi::PVView*) const
{
	return QString("New series view");
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
