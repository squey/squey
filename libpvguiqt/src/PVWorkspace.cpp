/**
 * \file PVWorkspace.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <QAction>
#include <QApplication>
#include <QHBoxLayout>
#include <QMenu>
#include <QPalette>
#include <QPushButton>
#include <QToolBar>
#include <QDateTime>

#include <pvkernel/core/PVDataTreeAutoShared.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>

#include <picviz/PVSource.h>
#include <picviz/PVView.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

#include <pvdisplays/PVDisplaysImpl.h>

#if 0
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#endif

#include <pvguiqt/PVLayerStackWidget.h>
#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVListingSortFilterProxyModel.h>
#include <pvguiqt/PVListingView.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVViewDisplay.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVProjectsTabWidget.h>


/******************************************************************************
 *
 * PVGuiQt::PVWorkspaceBase
 *
 *****************************************************************************/
uint64_t PVGuiQt::PVWorkspaceBase::_z_order_counter = 0;
bool PVGuiQt::PVWorkspaceBase::_drag_started = false;

PVGuiQt::PVWorkspaceBase::~PVWorkspaceBase() {}

PVGuiQt::PVWorkspaceBase* PVGuiQt::PVWorkspaceBase::workspace_under_mouse()
{
	QList<PVWorkspaceBase*> active_workspaces;
	for (QWidget* top_widget : QApplication::topLevelWidgets()) {
		QMainWindow* w = qobject_cast<QMainWindow*>(top_widget);
		if (w) {
			for (PVProjectsTabWidget* project_tab_widget : w->findChildren<PVProjectsTabWidget*>("PVProjectsTabWidget")) {
				PVWorkspacesTabWidget* workspace_tab_widget = qobject_cast<PVWorkspacesTabWidget*>(project_tab_widget->current_project());
				if (workspace_tab_widget) {
					PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(workspace_tab_widget->currentWidget());
					if (workspace) {
						active_workspaces.append(workspace);
					}
				}
			}
		}
	}

	assert(active_workspaces.size() > 0); // Hierarchy is supposed to be: PVMainWindow > PVProjectsTabWidget > PVWorkspaceTabWidget > PVWorkspaceBase.

	PVWorkspaceBase* workspace = nullptr;
	int z_order = -1;

	for (PVWorkspaceBase* w : active_workspaces) {
		if (w->geometry().contains(w->mapFromGlobal(QCursor::pos()))) {
			if (w->z_order() > z_order) {
				z_order = w->z_order();
				workspace = w;
			}
		}
	}

	return workspace;
}

void PVGuiQt::PVWorkspaceBase::changeEvent(QEvent *event)
{
	QMainWindow::changeEvent(event);

	if (event->type() == QEvent::ActivationChange && isActiveWindow()) {
		_z_order_index = ++_z_order_counter;
	}
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspaceBase::add_view_display(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_display /*= true*/, Qt::DockWidgetArea area /*= Qt::TopDockWidgetArea*/)
{
	PVViewDisplay* view_display = new PVViewDisplay(view, view_widget, name, can_be_central_display, this);

	connect(view_display, SIGNAL(destroyed(QObject*)), this, SLOT(display_destroyed(QObject*)));

	view_display->setWindowTitle(name);
	addDockWidget(area, view_display);
	connect(view_display, SIGNAL(try_automatic_tab_switch()), this, SLOT(emit_try_automatic_tab_switch()));

	_displays.append(view_display);

	return view_display;
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspaceBase::set_central_display(Picviz::PVView* view, QWidget* view_widget, const QString& name)
{
	PVViewDisplay* view_display = new PVViewDisplay(view, view_widget, name, true, this);
	view_display->setFeatures(QDockWidget::NoDockWidgetFeatures);
	view_display->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	setCentralWidget(view_display);

	_displays.append(view_display);

	return view_display;
}

void PVGuiQt::PVWorkspaceBase::switch_with_central_widget(PVViewDisplay* display_dock /* = nullptr */)
{
	if (!display_dock) {
		display_dock = (PVViewDisplay*) sender()->parent();
	}
	QWidget* display_widget = display_dock->widget();

	PVViewDisplay* central_dock = (PVViewDisplay*) centralWidget();

	if (central_dock) {
		QWidget* central_widget = central_dock->widget();

		// Exchange widgets
		central_dock->setWidget(display_widget);
		display_dock->setWidget(central_widget);

		// Exchange titles
		QString central_title = central_dock->windowTitle();
		central_dock->setWindowTitle(display_dock->windowTitle());
		display_dock->setWindowTitle(central_title);

		// Exchange colors
		QString style = QString("QDockWidget::title {background: %1;} QDockWidget { background: %2;} ");
		QColor col1 = central_dock->get_view()->get_color();
		QColor col2 = display_dock->get_view()->get_color();
		QString style1 = style.arg(col1.name()).arg(col1.name());
		QString style2 = style.arg(col2.name()).arg(col2.name());
		display_dock->setStyleSheet(style1);
		central_dock->setStyleSheet(style2);

		// Exchange views
		Picviz::PVView* central_view = central_dock->get_view();
		central_dock->set_view(display_dock->get_view());
		display_dock->set_view(central_view);
	}
	else {
		set_central_display(display_dock->get_view(), display_dock->widget(), display_dock->windowTitle());
		removeDockWidget(display_dock);
	}

}

void PVGuiQt::PVWorkspaceBase::display_destroyed(QObject* object /*= 0*/)
{
	PVGuiQt::PVViewDisplay* display = (PVGuiQt::PVViewDisplay*) object;
	_displays.removeAll(display);
}

/******************************************************************************
 *
 * PVGuiQt::PVWorkspace
 *
 *****************************************************************************/
PVGuiQt::PVWorkspace::PVWorkspace(Picviz::PVSource* source, QWidget* parent) :
	PVWorkspaceBase(parent),
	_source(source)
{
	//setTabPosition(Qt::TopDockWidgetArea, QTabWidget::North);

	_views_count = _source->get_children<Picviz::PVView>().size();

	// Register observers on the mapped and plotted
	source->depth_first_list(
		[&](PVCore::PVDataTreeObjectBase* o) {
			if(dynamic_cast<Picviz::PVMapped*>(o) || dynamic_cast<Picviz::PVPlotted*>(o)) {
				this->_obs.emplace_back(static_cast<QObject*>(this));
				datatree_obs_t* obs = &_obs.back();
				auto datatree_o = o->base_shared_from_this();
				PVHive::get().register_observer(datatree_o, *obs);
				obs->connect_refresh(this, SLOT(update_view_count(PVHive::PVObserverBase*)));
				obs->connect_about_to_be_deleted(this, SLOT(update_view_count(PVHive::PVObserverBase*)));
			}
		}
	);

	_toolbar = new QToolBar(this);
	_toolbar->setFloatable(false);
	_toolbar->setMovable(false);
	_toolbar->setIconSize(QSize(32, 32));
	addToolBar(_toolbar);

	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplaySourceIf>(
		[&](PVDisplays::PVDisplaySourceIf& obj)
		{
			QAction* act = PVDisplays::get().action_bound_to_params(obj, source);
			act->setCheckable(true);
			act->setIcon(obj.toolbar_icon());
			act->setToolTip(obj.tooltip_str());
			_toolbar->addAction(act);

			connect(act, SIGNAL(triggered()), this, SLOT(toggle_unique_source_widget()));
		}, PVDisplays::PVDisplayIf::ShowInToolbar & PVDisplays::PVDisplayIf::UniquePerParameters);

	_toolbar->addSeparator();

	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
		[&](PVDisplays::PVDisplayViewIf& obj)
		{
			if (!obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters)) {
				QToolButton* btn = new QToolButton(_toolbar);
				btn->setPopupMode(QToolButton::InstantPopup);
				btn->setIcon(obj.toolbar_icon());
				btn->setToolTip(obj.tooltip_str());
				_toolbar->addWidget(btn);

				_view_display_if_btns << std::make_pair(btn, &obj);
			}
		}, PVDisplays::PVDisplayIf::ShowInToolbar);

	_toolbar->addSeparator();

	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewAxisIf>(
		[&](PVDisplays::PVDisplayViewAxisIf& obj)
		{
			if (!obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters)) {
				QToolButton* btn = new QToolButton(_toolbar);
				btn->setPopupMode(QToolButton::InstantPopup);
				btn->setIcon(obj.toolbar_icon());
				btn->setToolTip(obj.tooltip_str());
				_toolbar->addWidget(btn);

				_view_axis_display_if_btns << std::make_pair(btn, &obj);
			}
		}, PVDisplays::PVDisplayIf::ShowInToolbar);

	_toolbar->addSeparator();

#if 0
	// Datatree views toolbar button
	_datatree_view_action = new QAction(_toolbar);
	_datatree_view_action->setCheckable(true);
	_datatree_view_action->setIcon(QIcon(":/view_display_datatree"));
	_datatree_view_action->setToolTip(tr("toggle data tree visibility"));
	connect(_datatree_view_action, SIGNAL(triggered(bool)), this, SLOT(show_datatree_view(bool)));
	_toolbar->addAction(_datatree_view_action);
	PVRootTreeModel* datatree_model = new PVRootTreeModel(*_source);
	PVRootTreeView* data_tree_view = new PVRootTreeView(datatree_model);
	PVGuiQt::PVViewDisplay* data_tree_view_display = add_view_display(nullptr, data_tree_view, "Data tree", false, Qt::RightDockWidgetArea);
	connect(data_tree_view_display, SIGNAL(display_closed()), this, SLOT(check_datatree_button()));
	check_datatree_button(true);

	// Layerstack views toolbar button
	_layerstack_tool_button = new QToolButton(_toolbar);
	_layerstack_tool_button->setPopupMode(QToolButton::InstantPopup);
	_layerstack_tool_button->setIcon(QIcon(":/layer-active.png"));
	_layerstack_tool_button->setToolTip(tr("Add layer stack"));
	_toolbar->addWidget(_layerstack_tool_button);
	_toolbar->addSeparator();

	// Listings button
	_listing_tool_button = new QToolButton(_toolbar);
	_listing_tool_button->setPopupMode(QToolButton::InstantPopup);
	_listing_tool_button->setIcon(QIcon(":/view_display_listing"));
	_listing_tool_button->setToolTip(tr("Add listing"));
	_toolbar->addWidget(_listing_tool_button);

	// Parallel views toolbar button
	_parallel_view_tool_button = new QToolButton(_toolbar);
	_parallel_view_tool_button->setPopupMode(QToolButton::InstantPopup);
	_parallel_view_tool_button->setIcon(QIcon(":/view_display_parallel"));
	_parallel_view_tool_button->setToolTip(tr("Add parallel view"));
	_toolbar->addWidget(_parallel_view_tool_button);

	// Zoomed parallel views toolbar button
	_zoomed_parallel_view_tool_button = new QToolButton(_toolbar);
	_zoomed_parallel_view_tool_button->setPopupMode(QToolButton::InstantPopup);
	_zoomed_parallel_view_tool_button->setIcon(QIcon(":/view_display_zoom"));
	_zoomed_parallel_view_tool_button->setToolTip(tr("Add zoomed parallel view"));
	_toolbar->addWidget(_zoomed_parallel_view_tool_button);

	// Scatter views toolbar button
	/*QToolButton* scatter_view_tool_button = new QToolButton(_toolbar);
	scatter_view_tool_button->setPopupMode(QToolButton::InstantPopup);
	scatter_view_tool_button->setIcon(QIcon(":/view_display_scatter"));
	scatter_view_tool_button->setToolTip(tr("Add scatter view"));
	_toolbar->addWidget(scatter_view_tool_button);*/
#endif

	refresh_views_menus();

#if 0
	for (auto view : _source->get_children<Picviz::PVView>()) {
		create_layerstack(view.get());
	}
#endif
}

void PVGuiQt::PVWorkspace::add_listing_view(bool central /*= false*/)
{
	QAction* action = (QAction*) sender();
	QVariant var = action->data();
	Picviz::PVView* view = var.value<Picviz::PVView*>();

	PVListingView* listing_view = create_listing_view(view->shared_from_this());

	QString title = "Listing [" + view->get_name() + "]";

	if (central) {
		set_central_display(view, listing_view, title);
	}
	else {
		add_view_display(view, listing_view, title);
	}
}

PVGuiQt::PVListingView* PVGuiQt::PVWorkspace::create_listing_view(Picviz::PVView_sp view_sp)
{
	PVListingModel* listing_model = new PVGuiQt::PVListingModel(view_sp);
	PVListingSortFilterProxyModel* proxy_model = new PVGuiQt::PVListingSortFilterProxyModel(view_sp);
	proxy_model->setSourceModel(listing_model);
	PVListingView* listing_view = new PVGuiQt::PVListingView(view_sp);
	listing_view->setModel(proxy_model);

	return listing_view;
}

void PVGuiQt::PVWorkspace::create_parallel_view(Picviz::PVView* view /*= nullptr*/)
{
#if 0
	if (!view) {
		QAction* action = (QAction*) sender();
		QVariant var = action->data();
		view = var.value<Picviz::PVView*>();
	}

	PVParallelView::PVLibView* parallel_lib_view;

	PVCore::PVProgressBox* pbox_lib = new PVCore::PVProgressBox("Creating new view...", (QWidget*) this);
	pbox_lib->set_enable_cancel(false);
	PVCore::PVProgressBox::progress<PVParallelView::PVLibView*>(boost::bind(&PVParallelView::common::get_lib_view, boost::ref(*view)), pbox_lib, parallel_lib_view);

	PVParallelView::PVFullParallelView* parallel_view = parallel_lib_view->create_view();
	connect(parallel_view, SIGNAL(new_zoomed_parallel_view(Picviz::PVView*, int)), this, SLOT(create_zoomed_parallel_view(Picviz::PVView*, int)));

	add_view_display(view, parallel_view, "Parallel view [" + view->get_name() + "]");
#endif
}

void PVGuiQt::PVWorkspace::create_zoomed_parallel_view()
{
#if 0
	QAction* action = (QAction*) sender();
	QVariant var = action->data();
	Picviz::PVView* view = var.value<Picviz::PVView*>();

	QDialog *dlg = new QDialog(this);
	dlg->setModal(true);

	QLayout *layout = new QVBoxLayout();
	dlg->setLayout(layout);

	QLabel *label = new QLabel("Open a zoomed view on axis:");
	layout->addWidget(label);

	PVWidgets::PVAxisIndexEditor *axes = new PVWidgets::PVAxisIndexEditor(*view, dlg);
	axes->set_axis_index(0);
	layout->addWidget(axes);

	QDialogButtonBox *dbb = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);

	QObject::connect(dbb, SIGNAL(accepted()), dlg, SLOT(accept()));
	QObject::connect(dbb, SIGNAL(rejected()), dlg, SLOT(reject()));

	layout->addWidget(dbb);

	if (dlg->exec() == QDialog::Accepted) {
		int axis_index = axes->get_axis_index().get_axis_index();
		create_zoomed_parallel_view(view, axis_index);

	}

	dlg->deleteLater();
#endif
}

void PVGuiQt::PVWorkspace::create_zoomed_parallel_view(Picviz::PVView* view, int axis_index)
{
#if 0
	QWidget* zoomed_parallel_view = PVParallelView::common::get_lib_view(*view)->create_zoomed_view(axis_index);
	add_view_display(view, zoomed_parallel_view, QString("Zoomed parallel view on axis '%1' [%2]").arg(view->get_axis_name(axis_index)).arg(view->get_name()));
#endif
}


void PVGuiQt::PVWorkspace::show_datatree_view(bool show)
{
	for (auto display : _displays) {
		if (dynamic_cast<PVRootTreeView*>(display->widget())) {
			display->setVisible(show);
		}
	}
}

void  PVGuiQt::PVWorkspace::create_layerstack(Picviz::PVView* view_org /*= nullptr*/)
{
	Picviz::PVView* view = nullptr;
	QAction* action;
	if (!view_org) {
		action = (QAction*) sender();
		QVariant var = action->data();
		view = var.value<Picviz::PVView*>();
	}
	else {
		view = view_org;
	}

	Picviz::PVView_sp view_sp = view->shared_from_this();
	PVLayerStackWidget* layerstack_view = new PVLayerStackWidget(view_sp);
	PVGuiQt::PVViewDisplay* layerstack_view_display = add_view_display(view, layerstack_view, "Layer stack [" + view->get_name() + "]", false, Qt::RightDockWidgetArea);
	connect(layerstack_view_display, SIGNAL(display_closed()), this, SLOT(destroy_layerstack()));

	if (!view_org) {
		action->setEnabled(false);
	}
	else {
		for (QAction* action : _layerstack_tool_button->actions()) {
			QVariant var = action->data();
			Picviz::PVView* view = var.value<Picviz::PVView*>();
			if (layerstack_view_display->get_view() == view) {
				action->setEnabled(false);
			}
		}
	}
}

void  PVGuiQt::PVWorkspace::destroy_layerstack()
{
	PVViewDisplay* view_display = (PVViewDisplay*) sender();

	for (QAction* action : _layerstack_tool_button->actions()) {
		QVariant var = action->data();
		Picviz::PVView* view = var.value<Picviz::PVView*>();
		if (view_display->get_view() == view) {
			action->setEnabled(true);
		}
	}
}

void PVGuiQt::PVWorkspace::check_datatree_button(bool check /*= false*/)
{
	_datatree_view_action->setChecked(check);
}

void PVGuiQt::PVWorkspace::update_view_count(PVHive::PVObserverBase* /*obs_base*/)
{
	uint64_t views_count = _source->get_children<Picviz::PVView>().size();
	if (views_count != _views_count) {
		refresh_views_menus();
		_views_count = views_count;
	}
}

const PVGuiQt::PVWorkspaceBase::PVViewWidgets& PVGuiQt::PVWorkspaceBase::get_view_widgets(Picviz::PVView* view)
{
	//assert(view->get_parent<Picviz::PVSource>() == _source);
	if (!_view_widgets.contains(view)) {
		PVViewWidgets widgets(view, this);
		return *(_view_widgets.insert(view, widgets));
	}
	return _view_widgets[view];
}

void PVGuiQt::PVWorkspace::toggle_unique_source_widget()
{
	QAction* act = qobject_cast<QAction*>(sender());
	if (!act) {
		return;
	}

	Picviz::PVSource* src = nullptr;
	PVDisplays::PVDisplaySourceIf& display_if = PVDisplays::get().get_params_from_action<PVDisplays::PVDisplaySourceIf>(*act, src);

	if (!src) {
		return;
	}

	QWidget* w = PVDisplays::get().get_widget(display_if, src);
	if (!w) {
		return;
	}

	PVViewDisplay* view_d = nullptr;
	for (PVViewDisplay* d: _displays) {
		if (d->widget() == w) {
			view_d = d;
			break;
		}
	}
	if (view_d) {
		view_d->setVisible(!view_d->isVisible());
	}
	else {
		add_view_display(nullptr, w, display_if.widget_title(src), display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget));
	}

}

void PVGuiQt::PVWorkspace::create_view_widget()
{
	QAction* act = qobject_cast<QAction*>(sender());
	if (!act) {
		return;
	}

	Picviz::PVView* view = nullptr;
	PVDisplays::PVDisplayViewIf& display_if = PVDisplays::get().get_params_from_action<PVDisplays::PVDisplayViewIf>(*act, view);

	if (!view) {
		return;
	}

	QWidget* w = PVDisplays::get().get_widget(display_if, view);
	add_view_display(view, w, display_if.widget_title(view), display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget));
}

void PVGuiQt::PVWorkspace::create_view_axis_widget()
{
	// All this should be the same than create_view_widget w/ a PVCore::PVArgumentList passed to create_widget
	QAction* act = qobject_cast<QAction*>(sender());
	if (!act) {
		return;
	}

	Picviz::PVView* view = nullptr;
	PVCol axis_comb = PVCOL_INVALID_VALUE;
	PVDisplays::PVDisplayViewAxisIf& display_if = PVDisplays::get().get_params_from_action<PVDisplays::PVDisplayViewAxisIf>(*act, view, axis_comb);

	if (!view) {
		return;
	}

	if (axis_comb == PVCOL_INVALID_VALUE) {
		PVCore::PVArgumentList args;
		args[PVCore::PVArgumentKey("axis", tr("New view on axis:"))].setValue(PVCore::PVAxisIndexType(0));
		if (!PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
		     PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*view),
		     args, this)) {
			return;
		}
		axis_comb = args["axis"].value<PVCore::PVAxisIndexType>().get_axis_index();
	}

	QWidget* w = PVDisplays::get().get_widget(display_if, view, axis_comb);
	add_view_display(view, w, display_if.widget_title(view, axis_comb), display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget));
}

void PVGuiQt::PVWorkspace::refresh_views_menus()
{
	for (std::pair<QToolButton*, PVDisplays::PVDisplayViewIf*> const& p: _view_display_if_btns) {
		for (QAction* act: p.first->actions()) {
			p.first->removeAction(act);
		}
	}
	for (std::pair<QToolButton*, PVDisplays::PVDisplayViewAxisIf*> const& p: _view_axis_display_if_btns) {
		for (QAction* act: p.first->actions()) {
			p.first->removeAction(act);
		}
	}

	for (Picviz::PVView_sp const& view: _source->get_children<Picviz::PVView>()) {
		QString action_name = view->get_name();

		for (std::pair<QToolButton*, PVDisplays::PVDisplayViewIf*> const& p: _view_display_if_btns) {
			QAction* act = PVDisplays::get().action_bound_to_params(*p.second, view.get());
			act->setText(action_name);
			p.first->addAction(act);

			connect(act, SIGNAL(triggered()), this, SLOT(create_view_widget()));
		}

		// AG: this category could go into PVDisplayViewIf w/ a PVCore::PVArgumentList object with one axis !
		for (std::pair<QToolButton*, PVDisplays::PVDisplayViewAxisIf*> const& p: _view_axis_display_if_btns) {
			QAction* act = PVDisplays::get().action_bound_to_params(*p.second, view.get(), PVCOL_INVALID_VALUE);
			act->setText(action_name);
			p.first->addAction(act);

			connect(act, SIGNAL(triggered()), this, SLOT(create_view_axis_widget()));
		}
	}

#if 0
	for (QAction* action : _layerstack_tool_button->actions()) {
		_layerstack_tool_button->removeAction(action);
	}
	for (QAction* action : _listing_tool_button->actions()) {
		_listing_tool_button->removeAction(action);
	}
	for (QAction* action : _parallel_view_tool_button->actions()) {
		_parallel_view_tool_button->removeAction(action);
	}
	for (QAction* action : _zoomed_parallel_view_tool_button->actions()) {
		_zoomed_parallel_view_tool_button->removeAction(action);
	}

	for (auto view : _source->get_children<Picviz::PVView>()) {

		QAction* action;
		QVariant var;
		var.setValue<Picviz::PVView*>(view.get());

		// Layer stack views menus
		action = new QAction(view->get_name(), this);
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_layerstack()));
		_layerstack_tool_button->addAction(action);

		// Listing views menus
		action = new QAction(view->get_name(), this);
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(add_listing_view()));
		_listing_tool_button->addAction(action);

		// Parallel views menus
		action = new QAction(view->get_name(), this);
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_parallel_view()));
		_parallel_view_tool_button->addAction(action);

		// Zoomed views menus
		action = new QAction(view->get_name(), this);
		action->setData(var);
		connect(action, SIGNAL(triggered(bool)), this, SLOT(create_zoomed_parallel_view()));
		_zoomed_parallel_view_tool_button->addAction(action);
	}
#endif
}


/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspace
 *
 *****************************************************************************/
PVGuiQt::PVOpenWorkspace::PVOpenWorkspace(QWidget* parent) : PVWorkspaceBase(parent)
{

}
