/**
 * \file PVWorkspacesTabWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PV_LISTINGS_TAB_WIDGET_H
#define PV_LISTINGS_TAB_WIDGET_H

#include <QTabWidget>

namespace PVInspector {

class PVMainWindow;
class PVTabSplitter;

/**
 * \class PVWorkspacesTabWidget
 */
class PVWorkspacesTabWidget : public QTabWidget
{
	Q_OBJECT

	PVMainWindow *main_window;

public:
	PVWorkspacesTabWidget(PVMainWindow *mw, QWidget *parent = 0);
	void remove_listing(PVTabSplitter* tab); // really used?

signals:
	void is_empty(); 

public slots:
	QTabBar *get_tabBar();
	void show_hide_views_tab_widget_Slot(bool visible);
	void tabCloseRequested_Slot(int index);
};

}

#endif // PV_LISTINGS_TAB_WIDGET_H
