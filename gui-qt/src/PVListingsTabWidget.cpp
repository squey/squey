//! \file PVListingsTabWidget.cpp
//! $Id: PVListingsTabWidget.cpp 2988 2011-05-26 10:30:25Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QWidget>
#include <QTabBar>

#include <pvkernel/core/general.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVListingsTabWidget.h>

/******************************************************************************
 *
 * PVInspector::PVListingsTabWidget::PVListingsTabWidget
 *
 *****************************************************************************/
PVInspector::PVListingsTabWidget::PVListingsTabWidget(PVMainWindow *mw, QWidget *parent) : QTabWidget(parent)
{
	main_window = mw;

	tabBar()->installEventFilter(main_window);

	setTabsClosable(true);
	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested_Slot(int)));
}


/******************************************************************************
 *
 * PVInspector::PVListingsTabWidget::get_tabBar
 *
 *****************************************************************************/
QTabBar *PVInspector::PVListingsTabWidget::get_tabBar()
{
	return tabBar();
}



/******************************************************************************
 *
 * PVInspector::PVListingsTabWidget::show_hide_views_tab_widget_Slot
 *
 *****************************************************************************/
void PVInspector::PVListingsTabWidget::show_hide_views_tab_widget_Slot(bool visible)
{
	setVisible(visible);
}

/******************************************************************************
 *
 * PVInspector::PVListingsTabWidget::tabCloseRequested_Slot
 *
 *****************************************************************************/
void PVInspector::PVListingsTabWidget::tabCloseRequested_Slot(int index)
{
	PVTabSplitter* tab = (PVTabSplitter*) widget(index);
	assert(tab);
	main_window->close_source(tab);
}

void PVInspector::PVListingsTabWidget::remove_listing(PVTabSplitter* tab)
{
	int index = indexOf(tab);
	assert(index != -1);
	removeTab(index);

	if(currentIndex() == -1)
	{
		emit is_empty();
		hide();
	}

	tab->deleteLater();
}
