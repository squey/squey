/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVStatsListingWidget.h>

#include <pvguiqt/PVDisplayViewListing.h>

#include <inendi/PVView.h>

#include <QObject>

PVDisplays::PVDisplayViewListing::PVDisplayViewListing()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                          PVDisplayIf::DefaultPresenceInSourceWorkspace,
                      "Listing",
                      Qt::NoDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewListing::create_widget(Inendi::PVView* view,
                                                         QWidget* parent) const
{
	Inendi::PVView_sp view_sp = view->shared_from_this();

	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view_sp);
	PVGuiQt::PVListingView* listing_view = new PVGuiQt::PVListingView(view_sp, parent);
	listing_view->setModel(model);

	PVGuiQt::PVHorizontalHeaderView* hheaderview =
	    new PVGuiQt::PVHorizontalHeaderView(Qt::Horizontal, listing_view);
	listing_view->setHorizontalHeader(hheaderview);

	PVGuiQt::PVStatsListingWidget* stats_listing = new PVGuiQt::PVStatsListingWidget(listing_view);

	return stats_listing;
}

QIcon PVDisplays::PVDisplayViewListing::toolbar_icon() const
{
	return QIcon(":/view-listing");
}

QString PVDisplays::PVDisplayViewListing::widget_title(Inendi::PVView* view) const
{
	return QString("Listing [" + view->get_name() + "]");
}
