/**
 * \file PVViewDisplay.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVVIEWDISPLAY_H__
#define __PVGUIQT_PVVIEWDISPLAY_H__

#include <QAction>
#include <QEvent>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFocusEvent>
#include <QContextMenuEvent>

#include <iostream>

namespace Picviz
{
class PVView;
}

namespace PVGuiQt
{

class PVWorkspaceBase;
class PVWorkspace;
class PVOpenWorkspace;
class DisplaysFocusInEventFilter;

class PVViewDisplay : public QDockWidget
{
	Q_OBJECT;

	friend PVWorkspaceBase;
	friend PVWorkspace;
	friend PVOpenWorkspace;

public:
	Picviz::PVView* get_view() { return _view; }
	void set_view(Picviz::PVView* view) { _view = view; }
	void set_current_view();

protected:
	bool event(QEvent* event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void closeEvent(QCloseEvent * event) override
	{
		emit display_closed();
		QDockWidget::closeEvent(event);
	}

public slots:
	void dragStarted(bool started);
	void dragEnded();

signals:
	void display_closed();
	void try_automatic_tab_switch();

private:
	void maximize_on_screen(int screen_number);

private:
	PVViewDisplay(Picviz::PVView* view, QWidget* view_widget, const QString& name, bool can_be_central_widget, PVWorkspaceBase* parent);

private:
	Picviz::PVView* _view;
	PVWorkspaceBase* _workspace;
	QPoint _press_pt;
};

}

#endif // #ifndef __PVGUIQT_PVVIEWDISPLAY_H__
