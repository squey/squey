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
#include <picviz/PVRoot_types.h>

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
#include <QTabBar>

namespace PVGuiQt
{

class PVStartScreenWidget;
class PVOpenWorkspacesWidget;
class PVSceneWorkspacesTabWidget;

namespace __impl
{

class PVTabBar : public QTabBar
{
	Q_OBJECT

public:
	PVTabBar(Picviz::PVRoot& root, QWidget* parent = 0) : QTabBar(parent), _root(root) {}

protected:
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent * event) override;

private slots:
	void rename_tab();
	void rename_tab(int index);

private:
	Picviz::PVRoot& _root;
};

class PVTabWidget : public QTabWidget
{
public:
	PVTabWidget(Picviz::PVRoot& root, QWidget* parent = 0) : QTabWidget(parent) { setTabBar(new PVTabBar(root, this)); }

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

/**
 * \class PVProjectsTabWidget
 *
 * \note This class is representing a project tab widget.
 */
class PVProjectsTabWidget : public QWidget
{
	Q_OBJECT

public:
	PVProjectsTabWidget(Picviz::PVRoot* root, QWidget* parent = 0);

	PVSceneWorkspacesTabWidget* add_project(Picviz::PVScene_p scene_p);
	void remove_project(PVSceneWorkspacesTabWidget* workspace_tab_widget);

	PVSourceWorkspace* add_source(Picviz::PVSource* source);

	void add_workspace(PVSourceWorkspace* workspace);
	void remove_workspace(PVSourceWorkspace* workspace, bool animation = true);

	bool save_modified_projects();
	bool is_current_project_untitled() { return current_project() ? current_project()->is_project_untitled() : false; }
	void collapse_tabs(bool collapse = true);

	inline Picviz::PVScene* current_scene() const { return _root->current_scene(); }
	PVWorkspacesTabWidgetBase* current_workspace_tab_widget() const;
	inline PVSceneWorkspacesTabWidget* current_project() const { return (_current_workspace_tab_widget_index >= 2) ? (PVSceneWorkspacesTabWidget*) _stacked_widget->widget(_current_workspace_tab_widget_index) : nullptr; }

	inline void select_tab_from_scene(Picviz::PVScene* scene);
	inline PVWorkspaceBase* current_workspace() const { return  current_project() ? (PVWorkspaceBase*) current_project()->currentWidget() : nullptr; }
	inline Picviz::PVView* current_view() const { return _root->current_view(); }
	inline int projects_count() { return _tab_widget->count() -2; }
	inline const QStringList get_projects_list()
	{
		QStringList projects_list;
		for (int i = 2; i < _tab_widget->count() ; i++) {
			projects_list << _tab_widget->tabText(i);
		}
		return projects_list;
	}
	inline int get_current_project_index() { return _current_workspace_tab_widget_index-2; }
	Picviz::PVScene* get_scene_from_path(const QString & path);
	PVSceneWorkspacesTabWidget* get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene);

private slots:
	void current_tab_changed(int index);
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }
	bool tab_close_requested(int index);
	void close_project();
	void project_modified(bool, QString = QString());
	void select_tab_from_current_scene();

signals:
	void is_empty();
	void workspace_dragged_outside(QWidget* workspace);
	void new_project();
	void load_source_from_description(PVRush::PVSourceDescription);
	void load_project();
	void load_project_from_path(const QString & project);
	void load_source();
	void import_type(const QString &);
	void new_format();
	void load_format();
	void edit_format(const QString & format);
	void save_project();

private:
	bool maybe_save_project(int index);
	void create_unclosable_tabs();
	void remove_project(int index);

private:
	__impl::PVSplitter* _splitter = nullptr;
	__impl::PVTabWidget* _tab_widget = nullptr; // QTabWidget has a problem with CSS and background-color, that's why this class isn't inheriting from QTabWidget...
	QStackedWidget* _stacked_widget = nullptr;
	PVStartScreenWidget* _start_screen_widget;
	PVOpenWorkspacesWidget* _workspaces_tab_widget;
	int _current_workspace_tab_widget_index;
	Picviz::PVRoot* _root;
};

}

#endif /* __PVGUIQT_PVPROJECTSTABWIDGET_H__ */
