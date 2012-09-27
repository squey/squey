/**
 * \file PVWorkspace.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACE_H__
#define __PVGUIQT_PVWORKSPACE_H__

#include <QMainWindow>
#include <QWidget>
#include <QList>
#include <QDockWidget>

namespace PVGuiQt
{

class PVWorkspace : public QMainWindow
{
public:
	PVWorkspace(QWidget* parent = 0);

	void add_view_display(QWidget* view_display, const QString& name);

private:
	QList<QDockWidget*> _displays;
};

}

#endif /* __PVGUIQT_PVWORKSPACE_H__ */
