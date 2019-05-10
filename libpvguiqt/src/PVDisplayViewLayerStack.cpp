/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>

#include <inendi/PVView.h>

PVDisplays::PVDisplayViewLayerStack::PVDisplayViewLayerStack()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                          PVDisplayIf::DefaultPresenceInSourceWorkspace,
                      "Layer stack",
                      QIcon(":/view-layerstack"),
                      Qt::RightDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewLayerStack::create_widget(Inendi::PVView* view,
                                                            QWidget* parent,
                                                            Params const&) const
{
	PVGuiQt::PVLayerStackWidget* widget = new PVGuiQt::PVLayerStackWidget(*view, parent);
	return widget;
}
