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
	_toolbar->setFloatable(false);
	_toolbar->setMovable(false);
	_toolbar->setIconSize(QSize(32, 32));
	addToolBar(_toolbar);

	// Datatree views toolbar button
	_datatree_view_action = new QAction(_toolbar);
	_datatree_view_action->setCheckable(true);
	_datatree_view_action->setIcon(QIcon(":/view_display_datatree"));
	_datatree_view_action->setToolTip(tr("Add data tree"));
	connect(_datatree_view_action, SIGNAL(triggered(bool)), this, SLOT(show_datatree_view(bool)));
	_toolbar->addAction(_datatree_view_action);
	_toolbar->addSeparator();
	PVRootTreeModel* datatree_model = new PVRootTreeModel(*_source);
	PVRootTreeView* data_tree_display = new PVRootTreeView(datatree_model);
	connect(data_tree_display, SIGNAL(destroyed(QObject*)), this, SLOT(check_datatree_button()));
	add_view_display(data_tree_display, "Data tree", false);
	check_datatree_button(true);

	// Listings button
	QToolButton* listing_tool_button = new QToolButton(_toolbar);
	listing_tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	listing_tool_button->setIcon(QIcon(":/view_display_listing"));
	listing_tool_button->setToolTip(tr("Add listing"));
	QMenu* listing_views_menu = new QMenu;
	for (auto view : source->get_children<Picviz::PVView>()) {
		QAction* action = new QAction(view->get_name(), this);
		QVariant var;
		var.setValue<Picviz::PVView*>(view.get());
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_listing_view()));
		listing_views_menu->addAction(action);
	}
	listing_tool_button->setMenu(listing_views_menu);
	_toolbar->addWidget(listing_tool_button);

	// Parallel views toolbar button
	QToolButton* parallel_view_tool_button = new QToolButton(_toolbar);
	parallel_view_tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	parallel_view_tool_button->setIcon(QIcon(":/view_display_parallel"));
	parallel_view_tool_button->setToolTip(tr("Add parallel view"));
	QMenu* parallel_views_menu = new QMenu;
	for (auto view : source->get_children<Picviz::PVView>()) {
		QAction* action = new QAction(view->get_name(), this);
		QVariant var;
		var.setValue<Picviz::PVView*>(view.get());
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_parallel_view()));
		parallel_views_menu->addAction(action);
	}
	parallel_view_tool_button->setMenu(parallel_views_menu);
	_toolbar->addWidget(parallel_view_tool_button);

	// Zoomed parallel views toolbar button
	QToolButton* zoomed_parallel_view_tool_button = new QToolButton(_toolbar);
	zoomed_parallel_view_tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	zoomed_parallel_view_tool_button->setIcon(QIcon(":/view_display_zoom"));
	zoomed_parallel_view_tool_button->setToolTip(tr("Add zoomed parallel view"));
	_toolbar->addWidget(zoomed_parallel_view_tool_button);

	// Scatter views toolbar button
	QToolButton* scatter_view_tool_button = new QToolButton(_toolbar);
	scatter_view_tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	scatter_view_tool_button->setIcon(QIcon(":/view_display_scatter"));
	scatter_view_tool_button->setToolTip(tr("Add scatter view"));
	_toolbar->addWidget(scatter_view_tool_button);
}

void PVGuiQt::PVWorkspace::add_view_display(QWidget* view_widget, const QString& name, bool can_be_central_display /*= true*/)
{
	PVViewDisplay* view_display = new PVViewDisplay(can_be_central_display, this);
	view_display->setWidget(view_widget);
	view_display->setWindowTitle(name);
	_displays.append(view_display);
	addDockWidget(Qt::TopDockWidgetArea, view_display);
}

void PVGuiQt::PVWorkspace::set_central_display(QWidget* view_widget, const QString& name)
{
	PVViewDisplay* view_display = new PVViewDisplay(true, this);
	view_display->setWidget(view_widget);
	view_display->setWindowTitle(name);
	_displays.append(view_display);
	view_display->setFeatures(QDockWidget::NoDockWidgetFeatures);
	setCentralWidget(view_display);
}
