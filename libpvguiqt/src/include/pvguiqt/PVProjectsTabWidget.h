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
#include <QTabWidget>
#include <QMouseEvent>
#include <QSize>

namespace PVGuiQt
{
class PVStartScreenWidget;

class PVProjectsTabWidget : public QTabWidget
{
	Q_OBJECT

public:
	PVProjectsTabWidget(QWidget* parent = 0);
	PVWorkspace* add_source(Picviz::PVSource* source);
	void add_workspace(PVWorkspace* workspace);
	void remove_workspace(PVWorkspace* workspace, bool animation = true);
	void remove_project(PVWorkspacesTabWidget* workspace_tab_widget);
	inline PVWorkspacesTabWidget* current_project() const { return _tab_widget->currentIndex() > 1 ? (PVWorkspacesTabWidget*) _tab_widget->currentWidget() : nullptr; }
	inline PVWorkspaceBase* current_workspace() const { return  current_project() ? (PVWorkspaceBase*) current_project()->currentWidget() : nullptr; }
	PVWorkspacesTabWidget* get_workspace_tab_widget_from_scene(const Picviz::PVScene* scene);

private slots:
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }
	void close_project();

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
	PVWorkspacesTabWidget* add_project(Picviz::PVScene* scene, const QString & text);
	void create_unclosable_tabs();

private:
	QTabWidget* _tab_widget = nullptr;

	PVStartScreenWidget* _start_screen_widget;
	PVWorkspacesTabWidget* _workspaces_tab_widget;
};

}

#endif /* __PVGUIQT_PVPROJECTSTABWIDGET_H__ */
