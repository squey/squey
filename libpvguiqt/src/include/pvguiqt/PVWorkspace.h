/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <iostream>

#include <QMainWindow>
#include <QWidget>
#include <QList>
#include <QDockWidget>
#include <QEvent>
#include <QPushButton>
#include <QHBoxLayout>

namespace PVGuiQt
{

class PVDockWidget : public QDockWidget
{
public:
	PVDockWidget(QWidget* parent = 0) : QDockWidget(parent) {}

	void setFlag(bool f) { flag = f; }

protected:
	/*bool event(QEvent* ev)
	{
		if (flag) {
			return QDockWidget::event(ev);
		}

		return false; // ?
	}*/

private:
	bool flag = true;
};

class PVWorkspace : public QMainWindow
{
	Q_OBJECT;
public:
	PVWorkspace(QWidget* parent = 0);

	void add_view_display(QWidget* view_display, const QString& name);

	void switch_with_central_widget(PVDockWidget* display_dock)
	{
		QWidget* display = display_dock->widget();
		display->setParent(nullptr);

		QWidget* central_widget = centralWidget();
		central_widget->setParent(display_dock);

		setCentralWidget(display);
		display_dock->setWidget(central_widget);
	}

private:
	QList<QDockWidget*> _displays;
};

class SlotHandler : public QObject
{
	Q_OBJECT;
public:
	SlotHandler(
		PVWorkspace* workspace,
		PVDockWidget* dock_display_to_switch
	) : _workspace(workspace), _dock_display(dock_display_to_switch) {}

public slots:
	void switch_displays()
	{
		_workspace->switch_with_central_widget(_dock_display);
	}

private:
	PVWorkspace* _workspace;
	PVDockWidget* _dock_display;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
