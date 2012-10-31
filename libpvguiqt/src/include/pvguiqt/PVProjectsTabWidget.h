/**
 * \file PVProjectsTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVPROJECTSTABWIDGET_H__
#define __PVGUIQT_PVPROJECTSTABWIDGET_H__

#include <assert.h>
#include <list>

#include <picviz/PVScene.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <QWidget>
#include <QObject>
#include <QTabWidget>
#include <QMouseEvent>
#include <QSize>
#include <QStackedWidget>
#include <QSplitterHandle>
#include <QSplitter>

namespace PVGuiQt
{

class PVStartScreenWidget;

namespace __impl
{

class PVTabWidget : public QTabWidget
{
public:
	PVTabWidget(QWidget* parent = 0) : QTabWidget(parent) {}

public:
	QTabBar* tabBar() const
	{
		return QTabWidget::tabBar();
	}
};

class PVSplitterHandle : public QSplitterHandle
{
public:
	PVSplitterHandle(Qt::Orientation orientation, QSplitter* parent = 0) : QSplitterHandle(orientation, parent) {}
	void set_max_size(int max_size) { _max_size = max_size; }
	int get_max_size() const { return _max_size; }

protected:
	void mouseMoveEvent(QMouseEvent* event) override
	{
		assert(_max_size > 0); // set splitter handle max size!
		QList<int> sizes = splitter()->sizes();
		assert(sizes.size() > 0);
		if ((sizes[0] == 0 && event->pos().x() < _max_size) || (sizes[0] != 0 && event->pos().x() < 0)) {
			QSplitterHandle::mouseMoveEvent(event);
		}
	}

private:
	int _max_size = 0;
};

class PVSplitter : public QSplitter
{
public:
	PVSplitter(Qt::Orientation orientation, QWidget * parent = 0) : QSplitter(orientation, parent) {}

protected:
    QSplitterHandle *createHandle()
    {
    	return new PVSplitterHandle(orientation(), this);
    }
};

}

class PVProjectsTabWidget : public QWidget
{
	Q_OBJECT

public:
	PVProjectsTabWidget(QWidget* parent = 0);
	PVWorkspacesTabWidget* add_project(Picviz::PVScene_p scene_p);
	void remove_project(PVWorkspacesTabWidget* workspace_tab_widget);
	PVWorkspace* add_source(Picviz::PVSource* source);
	void add_workspace(PVWorkspace* workspace);
	void remove_workspace(PVWorkspace* workspace, bool animation = true);
	bool save_modified_projects();
	void collapse_tabs(bool collapse = true);

	inline Picviz::PVScene* current_scene() const { return current_project()->get_scene(); }
	inline PVWorkspacesTabWidget* current_project() const { return (_current_project_index >= 2) ? (PVWorkspacesTabWidget*) _stacked_widget->widget(_current_project_index) : nullptr; }
	inline void select_project(Picviz::PVScene* scene) { _tab_widget->setCurrentIndex(_tab_widget->indexOf(get_workspace_tab_widget_from_scene(scene))); }
	inline void select_project(int index) { _tab_widget->setCurrentIndex(index+2); }
	inline PVWorkspaceBase* current_workspace() const { return  current_project() ? (PVWorkspaceBase*) current_project()->currentWidget() : nullptr; }
	inline Picviz::PVView* current_view() const { return current_project() ? current_project()->get_scene()->current_view() : nullptr; }
	inline int projects_count() { return _tab_widget->count() -2; }
	inline const QStringList get_projects_list()
	{
		QStringList projects_list;
		for (int i = 2; i < _tab_widget->count() ; i++) {
			projects_list << _tab_widget->tabText(i);
		}
		return projects_list;
	}
	inline int get_current_project_index() { return _current_project_index-2; }
	Picviz::PVScene* get_scene_from_path(const QString & path);
	PVWorkspacesTabWidget* get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene);

private slots:
	void current_tab_changed(int index);
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }
	bool tab_close_requested(int index);
	void close_project();
	void project_modified(bool, QString = QString());

signals:
	void is_empty();
	void workspace_dragged_outside(QWidget* workspace);
	void new_project();
	void load_source_from_description(PVRush::PVSourceDescription);
	void load_project();
	void load_project_from_path(const QString & project);
	void load_source();
	void new_format();
	void load_format();
	void edit_format(const QString & format);

private:
	bool maybe_save_project(int index);
	void create_unclosable_tabs();
	void remove_project(int index);

private:
	__impl::PVSplitter* _splitter = nullptr;
	__impl::PVTabWidget* _tab_widget = nullptr; // QTabWidget has a problem with CSS and background-color, that's why this class isn't inheriting from QTabWidget...
	QStackedWidget* _stacked_widget = nullptr;
	PVStartScreenWidget* _start_screen_widget;
	PVWorkspacesTabWidget* _workspaces_tab_widget;
	int _current_project_index;
};

}

#endif /* __PVGUIQT_PVPROJECTSTABWIDGET_H__ */
