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
    : PVDisplayViewAxisIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCtxtMenu,
                          "Timeseries view")
{
}

QWidget* PVDisplays::PVDisplayViewTimeseries::create_widget(Inendi::PVView* view,
                                                            PVCombCol axis_comb,
                                                            QWidget* parent) const
{
	return new PVParallelView::PVSeriesViewWidget(view, axis_comb, parent);
}

QIcon PVDisplays::PVDisplayViewTimeseries::toolbar_icon() const
{
	return QIcon(":/view-parallel-zoomed");
}

// FIXME : Hidden argument reflect bad design properties, inheritance should certainly be improved.
QString PVDisplays::PVDisplayViewTimeseries::widget_title(Inendi::PVView* view,
                                                          PVCombCol /*axis_comb*/) const
{
	return "Timeseries view [" + QString::fromStdString(view->get_name()) + "]";
}

QString PVDisplays::PVDisplayViewTimeseries::axis_menu_name(Inendi::PVView const* /*view*/,
                                                            PVCombCol /*axis_comb*/) const
{
	return QString("New timeseries view");
}

bool PVDisplays::PVDisplayViewTimeseries::should_add_to_menu(Inendi::PVView* view,
                                                             PVCombCol axis_comb)
{
	return view->get_axis(axis_comb).get_type().left(4) == "time";
}
