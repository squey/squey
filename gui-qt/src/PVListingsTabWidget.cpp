/**
 * \file PVListingsTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

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
	
	setObjectName("PVListingsTabWidget");

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
 * PVInspector::PVListingsTabWidget::remove_listing
 *
 *****************************************************************************/
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



