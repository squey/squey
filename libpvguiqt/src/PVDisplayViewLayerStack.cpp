/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>

#include <picviz/PVView.h>

PVDisplays::PVDisplayViewLayerStack::PVDisplayViewLayerStack():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget | PVDisplayIf::DefaultPresenceInSourceWorkspace, "Layer stack", Qt::RightDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewLayerStack::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	Picviz::PVView_sp view_sp = view->shared_from_this();
	PVGuiQt::PVLayerStackWidget* widget = new PVGuiQt::PVLayerStackWidget(view_sp, parent);
	return widget;
}

QIcon PVDisplays::PVDisplayViewLayerStack::toolbar_icon() const
{
	return QIcon(":/view-layerstack");
}

QString PVDisplays::PVDisplayViewLayerStack::widget_title(Picviz::PVView* view) const
{
	return QString("Layer stack [" + view->get_name() + "]");
}
