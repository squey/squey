#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVStatsListingWidget.h>

#include <pvguiqt/PVDisplayViewListing.h>

#include <picviz/PVView.h>

#include <QObject>

PVDisplays::PVDisplayViewListing::PVDisplayViewListing():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget | PVDisplayIf::DefaultPresenceInSourceWorkspace, "Listing", Qt::NoDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewListing::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	Picviz::PVView_sp view_sp = view->shared_from_this();

	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view_sp);
	PVGuiQt::PVListingView* listing_view = new PVGuiQt::PVListingView(view_sp, parent);
	PVGuiQt::PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view_sp, listing_view);
	proxy_model->setSourceModel(model);

	listing_view->setModel(proxy_model);
	PVGuiQt::PVStatsListingWidget* stats_listing = new PVGuiQt::PVStatsListingWidget(listing_view);

	return stats_listing;
}

QIcon PVDisplays::PVDisplayViewListing::toolbar_icon() const
{
	return QIcon(":/view_display_listing");
}

QString PVDisplays::PVDisplayViewListing::widget_title(Picviz::PVView* view) const
{
	return QString("Listing [" + view->get_name() + "]"); 
}
