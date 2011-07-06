//! \file PVListingsTabWidget.h
//! $Id: PVListingsTabWidget.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PV_LISTINGS_TAB_WIDGET_H
#define PV_LISTINGS_TAB_WIDGET_H

#include <QTabWidget>

namespace PVInspector {
class PVMainWindow;

/**
 * \class PVListingsTabWidget
 */
class PVListingsTabWidget : public QTabWidget
{
	Q_OBJECT

			PVMainWindow *main_window;

public:
	PVListingsTabWidget(PVMainWindow *mw, QWidget *parent = 0);

signals:
	void tabCloseRequested(int);

	public slots:
			QTabBar *get_tabBar();
	void show_hide_views_tab_widget_Slot(bool visible);
	void tabCloseRequested_Slot(int index);
};
}

#endif // PV_LISTINGS_TAB_WIDGET_H
