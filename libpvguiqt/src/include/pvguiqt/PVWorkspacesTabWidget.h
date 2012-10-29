/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <picviz/PVScene.h>

#include <pvkernel/core/lambda_connect.h>

#include <QApplication>
#include <QTabWidget>
#include <QTimer>
#include <QTabBar>
#include <QWidget>
#include <QMouseEvent>
#include <QPoint>
#include <QObject>
#include <QDrag>
#include <QLineEdit>
#include <QPropertyAnimation>

#include <iostream>

namespace Picviz
{
class PVSource;
class PVScene;
}

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVWorkspacesTabWidget;
class PVTabBar;

class TabRenamerEventFilter : public QObject
{
public:
	TabRenamerEventFilter(PVTabBar* tab_bar, int index, QLineEdit* line_edit) : _tab_bar(tab_bar), _index(index), _line_edit(line_edit) {}

	bool eventFilter(QObject* watched, QEvent* event);
private:
	PVTabBar* _tab_bar;
	int _index;
	QLineEdit* _line_edit;
};

class PVTabBar : public QTabBar
{
	Q_OBJECT

public:
	PVTabBar(PVWorkspacesTabWidget* tab_widget) : _tab_widget(tab_widget) {	connect(this, SIGNAL(currentChanged(int)), this, SLOT(tab_changed(int)));}
	QSize tabSizeHint(int index) const;

public:
	virtual void create_new_workspace() {}

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void leaveEvent(QEvent* even) override;

	void start_drag(QWidget* workspace);

protected slots:
	virtual void tab_changed(int index);

protected:
	PVWorkspacesTabWidget* _tab_widget;
	QPoint _drag_start_position;
	bool _drag_ongoing = false;
};

class PVWorkspaceTabBar : public PVTabBar
{
	Q_OBJECT

public:
	PVWorkspaceTabBar(PVWorkspacesTabWidget* tab_widget) : PVTabBar(tab_widget) {}
	int count() const;
	void create_new_workspace() override;

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

protected slots:
	void tab_changed(int index) override;
};

class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

	friend class PVTabBar;
	friend class PVWorkspaceTabBar;

public:
	PVWorkspacesTabWidget(Picviz::PVScene_p scene_p, QWidget* parent = 0);
	PVWorkspacesTabWidget(QWidget* parent = 0);
	void init();
	Picviz::PVScene* get_scene() { return _scene_p.get(); }
	void remove_workspace(int index, bool close_source = true);
	int addTab(PVWorkspaceBase* page, const QString & label);
	int count() const;

protected:
	void tabInserted(int index) override;
	void tabRemoved(int index) override;

signals:
	void workspace_dragged_outside(QWidget*);
	void workspace_closed(Picviz::PVSource* source);
	void is_empty();
	void animation_finished();

private slots:
	void tabCloseRequested_Slot(int index);
	void start_checking_for_automatic_tab_switch();
	void switch_tab();
	int get_tab_width() const { return 0; }
	void set_tab_width(int tab_width);
	void emit_workspace_dragged_outside(QWidget* workspace) { emit workspace_dragged_outside(workspace); }
	void animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state);

private:
	Picviz::PVScene_p _scene_p;
	QTimer _automatic_tab_switch_timer;
	int _tab_index;
	PVTabBar* _tab_bar;
	int _tab_animated_width;
	bool _tab_animation_ongoing = false;

	int _workspaces_count = 0;
	int _openworkspaces_count = 0;
};

}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
