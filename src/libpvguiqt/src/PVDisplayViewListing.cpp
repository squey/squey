//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVStatsListingWidget.h>

#include <pvguiqt/PVDisplayViewListing.h>

#include <squey/PVView.h>

#include <QObject>
#include <pvkernel/widgets/PVModdedIcon.h>

PVDisplays::PVDisplayViewListing::PVDisplayViewListing()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                      PVDisplayIf::DefaultPresenceInSourceWorkspace | PVDisplayIf::HasHelpPage,
                      "Listing",
                      PVModdedIcon("table-list"),
                      Qt::NoDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewListing::create_widget(Squey::PVView* view,
                                                         QWidget* parent,
                                                         Params const&) const
{
	auto* model = new PVGuiQt::PVListingModel(*view);
	auto* listing_view = new PVGuiQt::PVListingView(*view, parent);
	listing_view->setModel(model);

	auto* hheaderview =
	    new PVGuiQt::PVHorizontalHeaderView(Qt::Horizontal, listing_view);
	listing_view->setHorizontalHeader(hheaderview);

	auto* stats_listing = new PVGuiQt::PVStatsListingWidget(listing_view);
    stats_listing->setWindowTitle(default_window_title(*view));

	QObject::connect(listing_view, &PVGuiQt::PVListingView::set_status_bar_mouse_legend, [this,stats_listing](PVWidgets::PVMouseButtonsLegend legend) {
		_set_status_bar_mouse_legend.emit(stats_listing, legend);
	});
	QObject::connect(listing_view, &PVGuiQt::PVListingView::clear_status_bar_mouse_legend, [this,stats_listing](){
		_clear_status_bar_mouse_legend.emit(stats_listing);
	});

	return stats_listing;
}
