/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVWORKSPACESTABWIDGET_H__
#define __PVGUIQT_PVWORKSPACESTABWIDGET_H__

#include <QTabWidget>
#include <QWidget>

namespace PVGuiQt
{

class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT

public:
	PVWorkspacesTabWidget(QWidget* parent = 0);
};


}

#endif // __PVGUIQT_PVWORKSPACESTABWIDGET_H__
