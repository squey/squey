/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

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

class DragNDropTransparencyHack : public QObject
{
public:
	bool eventFilter(QObject* watched, QEvent* event);
};

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

class PVDrag : public QDrag
{
	Q_OBJECT

public:
	PVDrag(QWidget* drag_source) : QDrag(drag_source) {}

	~PVDrag()
	{
		if(!target()) {
			emit dragged_outside(QCursor::pos());
		}
	}

signals:
	void dragged_outside(QPoint pt);
};

class PVTabBar : public QTabBar
{
	Q_OBJECT

public:
	PVTabBar(PVWorkspacesTabWidget* tab_widget) : _tab_widget(tab_widget) {}
	QSize tabSizeHint(int index) const;
	int count() const;

protected:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void leaveEvent(QEvent* even) override;
	void wheelEvent(QWheelEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

public slots:
	void dragged_outside(QPoint);

private:
	void start_drag(QWidget* workspace);
	void stop_drag();

private:
	PVWorkspacesTabWidget* _tab_widget;
	QPoint _drag_start_position;
	bool _drag_ongoing = false;
};

class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT
	Q_PROPERTY(int tab_width READ get_tab_width WRITE set_tab_width);

	friend class PVTabBar;

public:
	PVWorkspacesTabWidget(Picviz::PVScene* scene, QWidget* parent = 0);
	void set_scene(Picviz::PVScene* scene) { _scene = scene; }
	void remove_workspace(int index);
	int addTab(PVWorkspaceBase* page, const QString & label, bool animation = true);
	int count() const;

protected:
	void tabInserted(int index) override;

signals:
	void workspace_dragged_outside(QPoint);
	void workspace_closed(Picviz::PVSource* source);
	void is_empty();

private slots:
	void tabCloseRequested_Slot(int index);
	void start_checking_for_automatic_tab_switch();
	void switch_tab();
	void tab_changed(int index);
	int get_tab_width() const { return 0; }
	void set_tab_width(int tab_width);
	void emit_workspace_dragged_outside(QPoint pt) { emit workspace_dragged_outside(pt); }
	void animation_state_changed(QAbstractAnimation::State new_state, QAbstractAnimation::State old_state);

private:
	Picviz::PVScene* _scene = nullptr;
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
