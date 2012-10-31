/**
 * \file PVProjectsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVStartScreenWidget.h>

#include <QHBoxLayout>

PVGuiQt::PVProjectsTabWidget::PVProjectsTabWidget(QWidget* parent /*= 0*/) : QWidget(parent)
{
	setObjectName("PVProjectsTabWidget");

	QHBoxLayout* main_layout = new QHBoxLayout();

	_tab_widget = new __impl::PVTabWidget();
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
	_workspaces_tab_widget = new PVWorkspacesTabWidget();
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

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::add_project(Picviz::PVScene_p scene_p)
{
	PVWorkspacesTabWidget* workspace_tab_widget = new PVWorkspacesTabWidget(scene_p);
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
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	QString text = _tab_widget->tabText(index);
	const QString star = "*";
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
		PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(i);
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
	PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) sender();
	assert(workspace_tab_widget);
	int index = _stacked_widget->indexOf(workspace_tab_widget);
	remove_project(index);
}

bool PVGuiQt::PVProjectsTabWidget::tab_close_requested(int index)
{
	if (maybe_save_project(index)) {
		remove_project(index);
		return true;
	}

	return false;
}

bool PVGuiQt::PVProjectsTabWidget::maybe_save_project(int index)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(index);
	if (tab_widget->is_project_modified()) {
		QMessageBox::StandardButton ret;
		QString project_name = _tab_widget->tabText(index).left(_tab_widget->tabText(index).size()-1);
		ret = QMessageBox::warning(this, tr("%1").arg(project_name),
				tr("The project \"%1\"has been modified.\n"
					"Do you want to save your changes?").arg(project_name),
				QMessageBox::Save | QMessageBox::Discard
				| QMessageBox::Cancel);
		if (ret == QMessageBox::Save) {
			emit save_project();
		}
		if (ret == QMessageBox::Discard) {
			return true;
		}
		else if (ret == QMessageBox::Cancel) {
			return false;
		}
	}
	return true;
#else
	return false;
#endif
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
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);

	if (!workspace_tab_widget) {
		workspace_tab_widget = add_project(scene->shared_from_this());
	}

	workspace_tab_widget->addTab(workspace, workspace->get_source()->get_name());
	_tab_widget->setCurrentIndex(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_workspace(PVWorkspace* workspace, bool animation /* = true */)
{
	Picviz::PVScene* scene = workspace->get_source()->get_parent<Picviz::PVScene>();
	PVWorkspacesTabWidget* workspace_tab_widget = get_workspace_tab_widget_from_scene(scene);
	workspace_tab_widget->remove_workspace(workspace_tab_widget->indexOf(workspace), animation);
}

void PVGuiQt::PVProjectsTabWidget::remove_project(PVWorkspacesTabWidget* workspace_tab_widget)
{
	remove_project(_stacked_widget->indexOf(workspace_tab_widget));
}

void PVGuiQt::PVProjectsTabWidget::remove_project(int index)
{
	if (index != -1) {
		PVWorkspacesTabWidget* tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(index);
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
	if (index >= 2) {
		_current_project_index = index;
	}
}

PVGuiQt::PVWorkspacesTabWidget* PVGuiQt::PVProjectsTabWidget::get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene)
{
	for (int i = 2 ; i < _stacked_widget->count(); i++) {
		PVWorkspacesTabWidget* workspace_tab_widget = (PVWorkspacesTabWidget*) _stacked_widget->widget(i);
		if (workspace_tab_widget->get_scene() == scene) {
			return workspace_tab_widget;
		}
	}
	return nullptr;
}

Picviz::PVScene* PVGuiQt::PVProjectsTabWidget::get_scene_from_path(const QString & path)
{
	for (int i = 2 ; i < _stacked_widget->count(); i++) {
		Picviz::PVScene* scene = ((PVWorkspacesTabWidget* ) _stacked_widget->widget(i))->get_scene();
		if (scene->get_path() == path) {
			return scene;
		}
	}

	return nullptr;
}
