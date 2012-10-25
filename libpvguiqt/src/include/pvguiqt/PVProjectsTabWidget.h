/**
 * \file PVProjectsTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVPROJECTSTABWIDGET_H__
#define __PVGUIQT_PVPROJECTSTABWIDGET_H__

#include <assert.h>

#include <picviz/PVScene.h>

#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVWorkspace.h>

#include <QWidget>
#include <QObject>
#include <QSplitter>
#include <QSplitterHandle>
#include <QTabBar>
#include <QMouseEvent>
#include <QStackedWidget>

namespace PVGuiQt
{
class PVStartScreenWidget;
class PVProjectsTabWidget;

namespace __impl
{

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
	PVWorkspace* add_source(Picviz::PVSource* source);
	void add_workspace(PVWorkspace* workspace);
	void remove_workspace(PVWorkspace* workspace, bool animation = true);
	void remove_project(PVWorkspacesTabWidget* workspace_tab_widget);
	void collapse_tabs(bool collapse = true);
	inline PVWorkspacesTabWidget* current_project() const { return (PVWorkspacesTabWidget*) _stacked_widget->currentWidget(); }
	inline PVWorkspaceBase* current_workspace() const { return  current_project() ? (PVWorkspaceBase*) current_project()->currentWidget() : nullptr; }
	PVWorkspacesTabWidget* get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene);

private slots:
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }
	void close_project();

signals:
	void is_empty();
	void workspace_dragged_outside(QWidget* workspace);

private:
	PVWorkspacesTabWidget* add_project(Picviz::PVScene* scene, const QString & text);
	void create_unclosable_tabs();

private:
	__impl::PVSplitter* _splitter = nullptr;
	QTabBar* _tab_bar = nullptr;
	QStackedWidget* _stacked_widget = nullptr;

	PVStartScreenWidget* _start_screen_widget;
	PVWorkspacesTabWidget* _workspaces_tab_widget;
};

}

#endif /* __PVGUIQT_PVPROJECTSTABWIDGET_H__ */
