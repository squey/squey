/**
 * \file PVWorkspace.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QAction>
#include <QToolButton>
#include <QMenu>

#include <pvkernel/core/PVDataTreeAutoShared.h>
#include <pvguiqt/PVWorkspace.h>
#include <picviz/PVView.h>

PVGuiQt::PVWorkspace::PVWorkspace(Picviz::PVSource_sp source, QWidget* parent) :
	QMainWindow(parent),
	_source(source)
{
	setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	_toolbar = new QToolBar(this);
	addToolBar(_toolbar);

	// Parallel views toolbar button
	QToolButton* tool_button = new QToolButton(_toolbar);
	tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	tool_button->setIcon(QIcon("/home/jbleonesio/dev/picviz-inspector/gui-qt/src/resources/inspector.png")); //tool_button->setIcon(QIcon(":/logo_text.png"));
	tool_button->setToolTip(tr("Add parallel view"));
	QMenu* views_menu = new QMenu;
	for (auto view : source->get_children<Picviz::PVView>()) {
		QAction* action = new QAction(view->get_name(), this);
		QVariant var;
		var.setValue<Picviz::PVView*>(view.get());
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_parallel_view()));
		views_menu->addAction(action);
	}
	tool_button->setMenu(views_menu);
	_toolbar->addWidget(tool_button);
}

void PVGuiQt::PVWorkspace::add_view_display(QWidget* view_widget, const QString& name)
{
	PVViewDisplay* view_display = new PVViewDisplay(this);
	view_display->setWidget(view_widget);
	view_display->setWindowTitle(name);

	addDockWidget(Qt::TopDockWidgetArea, view_display);
	_displays.append(view_display);
}
