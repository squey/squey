/**
 * \file PVWorkspace.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QAction>
#include <QHBoxLayout>
#include <QMenu>
#include <QPalette>
#include <QPushButton>
#include <QToolBar>

#include <pvkernel/core/PVDataTreeAutoShared.h>
#include <pvkernel/core/PVProgressBox.h>

#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVViewDisplay.h>

PVGuiQt::PVWorkspace::PVWorkspace(Picviz::PVSource_sp source, QWidget* parent) :
	QMainWindow(parent),
	_source(source)
{
	//setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	_toolbar = new QToolBar(this);
	_toolbar->setFloatable(false);
	_toolbar->setMovable(false);
	_toolbar->setIconSize(QSize(32, 32));
	addToolBar(_toolbar);

	// Datatree views toolbar button
	_datatree_view_action = new QAction(_toolbar);
	_datatree_view_action->setCheckable(true);
	_datatree_view_action->setIcon(QIcon(":/view_display_datatree"));
	_datatree_view_action->setToolTip(tr("toggle data tree visibility"));
	connect(_datatree_view_action, SIGNAL(triggered(bool)), this, SLOT(show_datatree_view(bool)));
	_toolbar->addAction(_datatree_view_action);
	PVRootTreeModel* datatree_model = new PVRootTreeModel(*_source);
	PVRootTreeView* data_tree_view = new PVRootTreeView(datatree_model);
	PVGuiQt::PVViewDisplay* data_tree_view_display = add_view_display(data_tree_view, "Data tree", false);
	connect(data_tree_view_display, SIGNAL(display_closed()), this, SLOT(check_datatree_button()));
	check_datatree_button(true);


	// Layerstack views toolbar button
	_layerstack_tool_button = new QToolButton(_toolbar);
	_layerstack_tool_button->setPopupMode(QToolButton::MenuButtonPopup);
	_layerstack_tool_button->setIcon(QIcon(":/layer-active.png"));
	_layerstack_tool_button->setToolTip(tr("Add layer stack"));
	for (auto view : source->get_children<Picviz::PVView>()) {
		PVLayerStackWidget* layerstack_view = new PVLayerStackWidget(view);
		PVGuiQt::PVViewDisplay* layerstack_view_display = add_view_display(layerstack_view, "Layer stack [" + view->get_name() + "]", false);
		QAction* action = new QAction(view->get_name(), layerstack_view_display);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(show_layerstack()));
		_layerstack_tool_button->addAction(action);
		connect(layerstack_view_display, SIGNAL(display_closed()), this, SLOT(hide_layerstack()));
		layerstack_view_display->setVisible(false); //
	}
	_toolbar->addWidget(_layerstack_tool_button);
	_toolbar->addSeparator();

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

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspace::add_view_display(QWidget* view_widget, const QString& name, bool can_be_central_display /*= true*/)
{
	PVViewDisplay* view_display = new PVViewDisplay(can_be_central_display, this);

    /*QPalette pal = view_display->palette();
    pal.setColor(QPalette::Background, QColor(50, 200, 1));
    view_display->setPalette(pal);
    view_display->setAutoFillBackground(true);*/

	//view_display->setStyleSheet("QDockWidget::title {background: purple;} QDockWidget { background: purple;} ");

	connect(view_display, SIGNAL(destroyed(QObject*)), this, SLOT(display_destroyed(QObject*)));
	view_display->setWidget(view_widget);
	view_display->setWindowTitle(name);
	addDockWidget(Qt::TopDockWidgetArea, view_display);

	_displays.append(view_display);

	return view_display;
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspace::set_central_display(QWidget* view_widget, const QString& name)
{
	PVViewDisplay* view_display = new PVViewDisplay(true, this);
	view_display->setWidget(view_widget);
	view_display->setWindowTitle(name);
	view_display->setFeatures(QDockWidget::NoDockWidgetFeatures);
	setCentralWidget(view_display);

	_displays.append(view_display);

	return view_display;
}

void PVGuiQt::PVWorkspace::switch_with_central_widget(PVViewDisplay* display_dock /*= nullptr*/)
{
	if (!display_dock) {
		display_dock = (PVViewDisplay*) sender()->parent();
	}
	QWidget* display_widget = display_dock->widget();

	PVViewDisplay* central_dock = (PVViewDisplay*) centralWidget();
	QWidget* central_widget = central_dock->widget();

	// Exchange widgets
	central_dock->setWidget(display_widget);
	display_dock->setWidget(central_widget);

	// Exchange titles
	QString central_title = central_dock->windowTitle();
	central_dock->setWindowTitle(display_dock->windowTitle());
	display_dock->setWindowTitle(central_title);
}

void PVGuiQt::PVWorkspace::create_listing_view()
{
	QAction* action = (QAction*) sender();
	QVariant var = action->data();
	Picviz::PVView* view = var.value<Picviz::PVView*>();

	Picviz::PVView_p view_p = view->shared_from_this();
	PVListingModel* listing_model = new PVGuiQt::PVListingModel(view_p);
	PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view_p);
	proxy_model->setSourceModel(listing_model);
	PVListingView* listing_view = new PVGuiQt::PVListingView(view_p);
	listing_view->setModel(proxy_model);

	add_view_display(listing_view, "Listing [" + view->get_name() + "]");
}

void PVGuiQt::PVWorkspace::create_parallel_view()
{
	QAction* action = (QAction*) sender();
	QVariant var = action->data();
	Picviz::PVView* view = var.value<Picviz::PVView*>();

	PVParallelView::PVLibView* parallel_lib_view;

	PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
	pbox_lib->set_enable_cancel(false);
	PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*view)), pbox_lib, parallel_lib_view);

	PVParallelView::PVFullParallelView* parallel_view = parallel_lib_view->create_view();
	connect(parallel_view, SIGNAL(new_zoomed_parallel_view(Picviz::PVView*, int)), this, SLOT(create_zoomed_parallel_view(Picviz::PVView*, int)));

	add_view_display(parallel_view, "Parallel view [" + view->get_name() + "]");
}

void PVGuiQt::PVWorkspace::create_zoomed_parallel_view(Picviz::PVView* view, int axis_id)
{
	QWidget* zoomed_parallel_view = PVParallelView::common::get_lib_view(*view)->create_zoomed_view(axis_id);

	add_view_display(zoomed_parallel_view, "Zoomed parallel view [" + view->get_name() + "]");
}

void PVGuiQt::PVWorkspace::show_datatree_view(bool show)
{
	for (auto display : _displays) {
		if (dynamic_cast<PVRootTreeView*>(display->widget())) {
			display->setVisible(show);
		}
	}
}

void  PVGuiQt::PVWorkspace::show_layerstack()
{
	QAction* action = (QAction*) sender();
	PVViewDisplay* view_display = (PVViewDisplay*) action->parent();
	view_display->setVisible(true);
	action->setEnabled(false);
}

void  PVGuiQt::PVWorkspace::hide_layerstack()
{
	PVViewDisplay* view_display = (PVViewDisplay*) sender();

	for (QAction* action : _layerstack_tool_button->actions()) {
		if (action->parent() == view_display) {
			action->setEnabled(true);
		}
	}
}

void PVGuiQt::PVWorkspace::check_datatree_button(bool check /*= false*/)
{
	_datatree_view_action->setChecked(check);
}

void PVGuiQt::PVWorkspace::display_destroyed(QObject* object /*= 0*/)
{
	PVGuiQt::PVViewDisplay* display = (PVGuiQt::PVViewDisplay*) object;
	_displays.removeAll(display);
}
