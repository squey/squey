/**
 * \file PVWorkspace.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVWorkspace.h>

PVGuiQt::PVWorkspace::PVWorkspace(QWidget* parent) : QMainWindow(parent)
{
	setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);
}

void PVGuiQt::PVWorkspace::add_view_display(QWidget* view_display, const QString& name)
{
	QDockWidget* view_display_dock_widget = new QDockWidget();
	view_display_dock_widget->setWidget(view_display);
	view_display_dock_widget->setWindowTitle(name);

	addDockWidget(Qt::TopDockWidgetArea, view_display_dock_widget);
	_displays.append(view_display_dock_widget);
}
