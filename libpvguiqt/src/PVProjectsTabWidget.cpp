/**
 * \file PVProjectsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVOpenWorkspacesWidget.h>
#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVStartScreenWidget.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVCallHelper.h>

#include <QHBoxLayout>
#include <QInputDialog>
#include <QMenu>
#include <QAction>

const QString star = "*";

/******************************************************************************
 *
 * PVGuiQt::__impl::PVTabBar
 *
 *****************************************************************************/
void PVGuiQt::__impl::PVTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
	int index = tabAt(event->pos());
	rename_tab(index);
	QTabBar::mouseDoubleClickEvent(event);
}

void PVGuiQt::__impl::PVTabBar::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::RightButton) {
		int index = tabAt(event->pos());
		QMenu* menu = new QMenu(this);
		QAction* rename_action = menu->addAction("&Rename");
		rename_action->setData(qVariantFromValue(index));
		connect(rename_action, SIGNAL(triggered(bool)), this, SLOT(rename_tab()));
		menu->popup(event->globalPos());
	}
	QTabBar::mousePressEvent(event);
}

void PVGuiQt::__impl::PVTabBar::keyPressEvent(QKeyEvent * event)
{
	if (event->key() == Qt::Key_F2) {
		rename_tab(currentIndex());
	}
	QTabBar::keyPressEvent(event);
}

void PVGuiQt::__impl::PVTabBar::rename_tab()
{
	QAction* rename_action = (QAction*) sender();
	assert(rename_action);
	int index = rename_action->data().toInt();
	rename_tab(index);
}

void PVGuiQt::__impl::PVTabBar::rename_tab(int index)
{
	QString tab_name = tabText(index);
	bool add_star = false;
	if (tab_name.endsWith(star)) {
		add_star = true;
		tab_name = tab_name.left(tab_name.size()-1);
	}
	QString name = QInputDialog::getText(this, "Rename data collection", "New data collection name:", QLineEdit::Normal, tab_name);
	if (!name.isEmpty()) {
		setTabText(index, name + (add_star ? star : ""));
		Picviz::PVScene* scene = _root.current_scene();
		assert(scene);
		Picviz::PVScene_p scene_p = scene->shared_from_this();
		PVHive::call<FUNC(Picviz::PVScene::set_name)>(scene_p, name);
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVProjectsTabWidget
 *
 *****************************************************************************/
PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(Picviz::PVRoot& root, QWidget* parent /*= 0*/):
	QWidget(parent),
	_root(root)
{
	setObjectName("PVProjectsTabWidget");

	QHBoxLayout* main_layout = new QHBoxLayout();

	_tab_widget = new __impl::PVTabWidget(root);
	_tab_widget->setTabsClosable(true);

	_stacked_widget = new QStackedWidget();

	create_unclosable_tabs();

	_splitter = new __impl::PVSplitter(Qt::Horizontal);
	_splitter->setChildrenCollapsible(true);
	_splitter->addWidget(_tab_widget);
	_splitter->addWidget(_stacked_widget);
	_splitter->setStretchFactor(0, 0);
	_splitter->setStretchFactor(1, 1);
	int tab_width = _tab_widget->tabBar()->tabRect(0).width();
	((__impl::PVSplitterHandle*) _splitter->handle(1))->set_max_size(tab_width);
	QList<int> sizes;
	sizes << 1 << 2;
	_splitter->setSizes(sizes);

	main_layout->addWidget(_splitter);

	setLayout(main_layout);

	connect(_tab_widget, SIGNAL(currentChanged(int)), this, SLOT(current_tab_changed(int)));
	connect(_tab_widget->tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tab_close_requested(int)));

	// Hive
	// Register for current scene changing
	Picviz::PVRoot_sp root_sp = root.shared_from_this();
	PVHive::PVObserverSignal<Picviz::PVRoot>* obs = new PVHive::PVObserverSignal<Picviz::PVRoot>(this);
	obs->connect_refresh(this, SLOT(select_tab_from_current_scene()));
	PVHive::get().register_observer(root_sp, [=](Picviz::PVRoot& root) { return root.get_current_scene_hive_property(); }, *obs);
}

void  PVGuiQt::PVProjectsTabWidget::create_unclosable_tabs()
{
	// Start screen widget
	_start_screen_widget = new PVGuiQt::PVStartScreenWidget();
	_tab_widget->addTab(new QWidget(), "");
	_tab_widget->setTabPosition(QTabWidget::West);
	_tab_widget->tabBar()->tabButton(0, QTabBar::RightSide)->resize(0, 0);
	_tab_widget->setTabIcon(0, QIcon(":/picviz"));
	_tab_widget->setTabToolTip(0, "Start screen");
	_stacked_widget->addWidget(_start_screen_widget);
	connect(_start_screen_widget, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)), this, SIGNAL(load_source_from_description(PVRush::PVSourceDescription)));
	connect(_start_screen_widget, SIGNAL(new_project()), this, SIGNAL(new_project()));
	connect(_start_screen_widget, SIGNAL(load_project()), this, SIGNAL(load_project()));
	connect(_start_screen_widget, SIGNAL(load_project_from_path(const QString &)), this, SIGNAL(load_project_from_path(const QString &)));
	connect(_start_screen_widget, SIGNAL(import_type(const QString &)), this, SIGNAL(import_type(const QString &)));
	connect(_start_screen_widget, SIGNAL(new_format()), this, SIGNAL(new_format()));
	connect(_start_screen_widget, SIGNAL(load_format()), this, SIGNAL(load_format()));
	connect(_start_screen_widget, SIGNAL(edit_format(const QString &)), this, SIGNAL(edit_format(const QString &)));

	// Open workspaces
	_workspaces_tab_widget = new PVOpenWorkspacesWidget(&_root);
	_tab_widget->addTab(new QWidget(), "");
	_tab_widget->tabBar()->tabButton(1, QTabBar::RightSide)->resize(0, 0);
	_tab_widget->setTabToolTip(1, "Workspaces");
	_tab_widget->setTabIcon(1, QIcon(":/brush.png"));
	_stacked_widget->addWidget(_workspaces_tab_widget);
}

void PVGuiQt::PVProjectsTabWidget::collapse_tabs(bool collapse /* = true */)
{
	int max_size = ((__impl::PVSplitterHandle*) _splitter->handle(1))->get_max_size();
	QList<int> sizes;
	sizes << (collapse ? 0 : max_size) << 1;
	_splitter->setSizes(sizes);
}

PVGuiQt::PVSceneWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene_p scene_p)
{
	PVSceneWorkspacesTabWidget* workspace_tab_widget = new PVSceneWorkspacesTabWidget(*scene_p);
	connect(workspace_tab_widget, SIGNAL(workspace_dragged_outside(QWidget*)), this, SLOT(emit_workspace_dragged_outside(QWidget*)));
	connect(workspace_tab_widget, SIGNAL(is_empty()), this, SLOT(close_project()));
	connect(workspace_tab_widget, SIGNAL(project_modified(bool, QString)), this, SLOT(project_modified(bool, QString)));

	int index = _tab_widget->count();
	_tab_widget->insertTab(index, new QWidget(), scene_p->get_name());
	_stacked_widget->insertWidget(index, workspace_tab_widget);
	_tab_widget->setTabToolTip(index, scene_p->get_path());
	_tab_widget->setCurrentIndex(index);

	return workspace_tab_widget;
}

void PVGuiQt::PVProjectsTabWidget::project_modified(bool modified, QString path /* = QString */)
{
	PVSceneWorkspacesTabWidget* workspace_tab_widget = (PVSceneWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	QString text = _tab_widget->tabText(index);
	if (modified && !text.endsWith(star)) {
		_tab_widget->setTabText(index, text + "*");
	}
	else if (!modified && text.endsWith(star)) {
		if (path.isEmpty()) {
			_tab_widget->setTabText(index, text.left(text.size()));
		}
		else {
			QFileInfo info(path);
			QString basename = info.fileName();
			_tab_widget->setTabToolTip(index, path);
			_tab_widget->setTabText(index, basename);
		}
	}
}

bool PVGuiQt::PVProjectsTabWidget::save_modified_projects()
{
	for (int i = 2; i < _tab_widget->count(); i++) {
		PVSceneWorkspacesTabWidget* tab_widget = (PVSceneWorkspacesTabWidget*) _stacked_widget->widget(i);
		if (tab_widget->is_project_modified()) {
			if (!tab_close_requested(i)) {
				return false;
			}
		}
	}

	return true;
}

void PVGuiQt::PVProjectsTabWidget::close_project()
{
	PVSceneWorkspacesTabWidget* workspace_tab_widget = (PVSceneWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	remove_project(index);
}

bool PVGuiQt::PVProjectsTabWidget::tab_close_requested(int index)
{
	remove_project(index);
	return true;
}

PVGuiQt::PVWorkspace* PVGuiQt::PVProjectsTabWidget::add_source(Picviz::PVSource* source)
{
	PVGuiQt::PVWorkspace* workspace = new PVGuiQt::PVWorkspace(source);

	add_workspace(workspace);

	return workspace;
}

void PVGuiQt::PVProjectsTabWidget::add_workspace(PVWorkspace* workspace)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVSceneWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);

	if (!workspace_tab_widget) {
		workspace_tab_widget = add_project(scene->shared_from_this());
	}

	workspace_tab_widget->addTab(workspace, workspace->get_source()->get_window_name());
	_tab_widget->setCurrentIndex(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_workspace(PVWorkspace* workspace, bool animation /* = true */)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVSceneWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);
	workspace_tab_widget->remove_workspace(workspace_tab_widget->indexOf(workspace), animation);
}

void PVGuiQt::PVProjectsTabWidget::remove_project(PVSceneWorkspacesTabWidget* workspace_tab_widget)
{
	remove_project(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_project(int index)
{
	if (index != -1) {
		PVSceneWorkspacesTabWidget* tab_widget = (PVSceneWorkspacesTabWidget*) _stacked_widget->widget(index);
		tab_widget->get_scene()->remove_from_tree();
		_tab_widget->removeTab(index);
		_stacked_widget->removeWidget(tab_widget);

		tab_widget->deleteLater();

		if (_tab_widget->count() == 2) {
			_tab_widget->setCurrentIndex(0);
			emit is_empty();
		}
	}
}

void PVGuiQt::PVProjectsTabWidget::current_tab_changed(int index)
{
	_stacked_widget->setCurrentIndex(index); // Map QTabBar signal to QStackedWidget to keep the sync
	_current_workspace_tab_widget_index = index;

	if (index == 0) {
		return;
	}

	QWidget* new_widget = _stacked_widget->widget(index);
	PVWorkspacesTabWidgetBase* workspace_tab_widget;

	Picviz::PVAD2GView* correlation = nullptr;

	if (index == 1) {
		PVOpenWorkspacesWidget* w = qobject_cast<PVOpenWorkspacesWidget*>(new_widget);
		assert(w);
		workspace_tab_widget = w->workspace_tab_widget();
		correlation = workspace_tab_widget->get_correlation();
	}
	else {
		workspace_tab_widget = qobject_cast<PVWorkspacesTabWidgetBase*>(new_widget);
		assert(workspace_tab_widget);
		correlation = workspace_tab_widget->get_correlation();

		PVSceneWorkspacesTabWidget* scene_tab = qobject_cast<PVSceneWorkspacesTabWidget*>(workspace_tab_widget);
		assert(scene_tab);

		Picviz::PVRoot_sp root_sp = _root.shared_from_this();
		PVHive::call<FUNC(Picviz::PVRoot::select_scene)>(root_sp, *scene_tab->get_scene());
	}

	_root.select_correlation(correlation);
}

PVGuiQt::PVWorkspacesTabWidgetBase* PVGuiQt::PVProjectsTabWidget::current_workspace_tab_widget() const
{
	if (_current_workspace_tab_widget_index < 0) {
		return nullptr;
	}

	QWidget* w = _stacked_widget->widget(_current_workspace_tab_widget_index);
	if (_current_workspace_tab_widget_index == 1) {
		return qobject_cast<PVOpenWorkspacesWidget*>(w)->workspace_tab_widget();
	}

	return qobject_cast<PVWorkspacesTabWidgetBase*>(w);
}

PVGuiQt::PVSceneWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene)
{
	for (int i = 2 ; i < _stacked_widget->count(); i++) {
		PVSceneWorkspacesTabWidget* workspace_tab_widget = (PVSceneWorkspacesTabWidget*) _stacked_widget->widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}

void PVGuiQt::PVProjectsTabWidget::select_tab_from_scene(Picviz::PVScene* scene)
{
	_tab_widget->setCurrentIndex(_tab_widget->indexOf(get_workspace_tab_widget_from_scene(scene)));
}

void PVGuiQt::PVProjectsTabWidget::select_tab_from_current_scene()
{
	Picviz::PVScene* cur_scene = _root.current_scene();
	if (cur_scene->get_parent<Picviz::PVRoot>() != &_root) {
		return;
	}

	select_tab_from_scene(cur_scene);
}
