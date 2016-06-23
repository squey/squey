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
                      Qt::RightDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewLayerStack::create_widget(Inendi::PVView* view,
                                                            QWidget* parent) const
{
	PVGuiQt::PVLayerStackWidget* widget = new PVGuiQt::PVLayerStackWidget(*view, parent);
	return widget;
}

QIcon PVDisplays::PVDisplayViewLayerStack::toolbar_icon() const
{
	return QIcon(":/view-layerstack");
}

QString PVDisplays::PVDisplayViewLayerStack::widget_title(Inendi::PVView* view) const
{
	return "Layer stack [" + QString::fromStdString(view->get_name()) + "]";
}
