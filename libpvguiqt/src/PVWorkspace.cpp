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
				PVWorkspacesTabWidgetBase* workspace_tab_widget = project_tab_widget->current_workspace_tab_widget();
				if (workspace_tab_widget) {
					PVWorkspaceBase* workspace = qobject_cast<PVWorkspaceBase*>(workspace_tab_widget->currentWidget());
					if (workspace) {
						active_workspaces.append(workspace);
					}
				}
			}
		}
	}

	if (active_workspaces.size() == 0) {
		return nullptr;
	}

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

void PVGuiQt::PVWorkspaceBase::displays_about_to_be_deleted()
{
	for (PVViewDisplay* display : _displays) {
		display->about_to_be_deleted();
	}
}

void PVGuiQt::PVWorkspaceBase::changeEvent(QEvent *event)
{
	QMainWindow::changeEvent(event);

	if (event->type() == QEvent::ActivationChange && isActiveWindow()) {
		_z_order_index = ++_z_order_counter;
	}
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspaceBase::add_view_display(Picviz::PVView* view, QWidget* view_widget, std::function<QString()> name, bool can_be_central_display /*= true*/, bool delete_on_close /* = true*/, Qt::DockWidgetArea area /*= Qt::TopDockWidgetArea*/)
{
	PVViewDisplay* view_display = new PVViewDisplay(view, view_widget, name, can_be_central_display, delete_on_close, this);

	connect(view_display, SIGNAL(destroyed(QObject*)), this, SLOT(display_destroyed(QObject*)));

	view_display->setWindowTitle(name());
	addDockWidget(area, view_display);
	connect(view_display, SIGNAL(try_automatic_tab_switch()), this, SIGNAL(try_automatic_tab_switch()));

	_displays.append(view_display);

	return view_display;
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspaceBase::set_central_display(Picviz::PVView* view, QWidget* view_widget, std::function<QString()> name, bool delete_on_close)
{
	PVViewDisplay* view_display = new PVViewDisplay(view, view_widget, name, true, delete_on_close, this);
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
		QColor col1 = central_dock->get_view()->get_color();
		QColor col2 = display_dock->get_view()->get_color();
		QPalette Pal1(display_dock->palette());
		Pal1.setColor(QPalette::Background, col1);
		display_dock->setAutoFillBackground(true);
		display_dock->setPalette(Pal1);
		QPalette Pal2(central_dock->palette());
		Pal2.setColor(QPalette::Background, col2);
		central_dock->setAutoFillBackground(true);
		central_dock->setPalette(Pal2);

		// Exchange name functions
		std::function<QString()> tmp_name;
		tmp_name = central_dock->_name;
		central_dock->_name = display_dock->_name;
		display_dock->_name = tmp_name;

		// Exchange views and view events registering
		Picviz::PVView* central_view = central_dock->get_view();
		Picviz::PVView* display_view = display_dock->get_view();
		central_dock->set_view(display_view);
		central_dock->register_view(display_view);
		display_dock->set_view(central_view);
		display_dock->register_view(central_view);
	}
	else {
		set_central_display(display_dock->get_view(), display_dock->widget(), display_dock->_name, display_dock->testAttribute(Qt::WA_DeleteOnClose));
		removeDockWidget(display_dock);
	}
}

void PVGuiQt::PVWorkspaceBase::display_destroyed(QObject* object /*= 0*/)
{
	PVGuiQt::PVViewDisplay* display = (PVGuiQt::PVViewDisplay*) object;
	_displays.removeAll(display);
}

void PVGuiQt::PVWorkspaceBase::toggle_unique_source_widget(QAction* act)
{
	// All this should be the same than create_view_widget w/ a PVCore::PVArgumentList passed to create_widget
	if (!act) {
		act = qobject_cast<QAction*>(sender());
		if (!act) {
			return;
		}
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
		add_view_display(nullptr, w, [&,src](){ return display_if.widget_title(src); }, display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget), false);
	}
}

void PVGuiQt::PVWorkspaceBase::create_view_widget(QAction* act)
{
	if (!act) {
		act = qobject_cast<QAction*>(sender());
		if (!act) {
			return;
		}
	}

	Picviz::PVView* view = nullptr;
	PVDisplays::PVDisplayViewIf& display_if = PVDisplays::get().get_params_from_action<PVDisplays::PVDisplayViewIf>(*act, view);

	if (!view) {
		return;
	}

	QWidget* w = PVDisplays::get().get_widget(display_if, view);
	add_view_display(view, w, [&,view](){ return display_if.widget_title(view); }, display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget), true);
}

void PVGuiQt::PVWorkspaceBase::create_view_axis_widget(QAction* act)
{
	// All this should be the same than create_view_widget w/ a PVCore::PVArgumentList passed to create_widget
	if (!act) {
		act = qobject_cast<QAction*>(sender());
		if (!act) {
			return;
		}
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
	add_view_display(view, w, [&,view,axis_comb](){ return display_if.widget_title(view, axis_comb); }, display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget), true);
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

	refresh_views_menus();

	for (Picviz::PVView_sp const& view: _source->get_children<Picviz::PVView>()) {
		bool already_center = false;
		// Create default widgets
		PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
			[&](PVDisplays::PVDisplayViewIf& obj)
			{
				QWidget* w = PVDisplays::get().get_widget(obj, view.get());

				Picviz::PVView* v = view.get();
				std::function<QString()> name = [&,v](){ return obj.widget_title(v); };
				const bool as_central = obj.default_position_as_central_hint();

				const bool delete_on_close = !obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters);
				if (as_central && !already_center) {
					set_central_display(view.get(), w, name, delete_on_close);
				}
				else {
					Qt::DockWidgetArea pos = obj.default_position_hint();
					if (as_central && already_center) {
						pos = Qt::TopDockWidgetArea;
					}
					add_view_display(view.get(), w, name, obj.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget), delete_on_close, pos);
				}
			}, PVDisplays::PVDisplayIf::DefaultPresenceInSourceWorkspace);
	}
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
}


/******************************************************************************
 *
 * PVGuiQt::PVOpenWorkspace
 *
 *****************************************************************************/
PVGuiQt::PVOpenWorkspace::PVOpenWorkspace(QWidget* parent) : PVWorkspaceBase(parent)
{

}
