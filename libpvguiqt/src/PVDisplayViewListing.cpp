#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>

#include <pvguiqt/PVDisplayViewListing.h>

#include <picviz/PVView.h>

PVDisplays::PVDisplayViewListing::PVDisplayViewListing():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar)
{
}

QWidget* PVDisplays::PVDisplayViewListing::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	Picviz::PVView_sp view_sp = view->shared_from_this();

	PVGuiQt::PVListingModel* model = new PVGuiQt::PVListingModel(view_sp);
	PVGuiQt::PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view_sp);
	proxy_model->setSourceModel(model);
	PVGuiQt::PVListingView* widget = new PVGuiQt::PVListingView(view_sp, parent);
	widget->setModel(proxy_model);

	return widget;
}

QIcon PVDisplays::PVDisplayViewListing::toolbar_icon() const
{
	return QIcon(":/view_display_listing");
}

QString PVDisplays::PVDisplayViewListing::widget_title(Picviz::PVView* view) const
{
	return QString("Listing [" + view->get_name() + "]"); 
}
