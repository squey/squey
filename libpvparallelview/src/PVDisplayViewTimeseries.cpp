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
    : PVDisplayViewDataIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu, "Series view")
{
}

QWidget* PVDisplays::PVDisplayViewTimeseries::create_widget(Inendi::PVView* view,
                                                            Params const& data,
                                                            QWidget* parent) const
{
	auto axis_comb = data.at(0);
	return new PVParallelView::PVSeriesViewWidget(view, axis_comb, parent);
}

QIcon PVDisplays::PVDisplayViewTimeseries::toolbar_icon() const
{
	return QIcon(":/view-series");
}

// FIXME : Hidden argument reflect bad design properties, inheritance should certainly be improved.
QString PVDisplays::PVDisplayViewTimeseries::widget_title(Inendi::PVView* view,
                                                          Params const& data) const
{
	auto axis_comb = data.at(0);
	return "Series view [" + QString::fromStdString(view->get_name()) + " on axis '" +
	       view->get_axis_name(axis_comb) + "']";
}

QString PVDisplays::PVDisplayViewTimeseries::axis_menu_name(Inendi::PVView*, Params const&) const
{
	return QString("New series view");
}

bool PVDisplays::PVDisplayViewTimeseries::should_add_to_menu(Inendi::PVView* view,
                                                             Params const& data) const
{
	auto axis_comb = data.at(0);
	return view->get_axis(axis_comb).get_type().left(4) == "time" or
	       view->get_axis(axis_comb).get_type().left(7) == "number_";
}
