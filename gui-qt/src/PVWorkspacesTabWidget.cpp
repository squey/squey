/**
 * \file PVWorkspacesTabWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QWidget>
#include <QTabBar>

#include <pvkernel/core/general.h>

#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <PVWorkspacesTabWidget.h>

/******************************************************************************
 *
 * PVInspector::PVWorkspacesTabWidget::PVWorkspacesTabWidget
 *
 *****************************************************************************/
PVInspector::PVWorkspacesTabWidget::PVWorkspacesTabWidget(PVMainWindow *mw, QWidget *parent) : QTabWidget(parent)
{
	main_window = mw;

	tabBar()->installEventFilter(main_window);
	
	setObjectName("PVWorkspacesTabWidget");

	setTabsClosable(true);
	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested_Slot(int)));
}


/******************************************************************************
 *
 * PVInspector::PVWorkspacesTabWidget::get_tabBar
 *
 *****************************************************************************/
QTabBar *PVInspector::PVWorkspacesTabWidget::get_tabBar()
{
	return tabBar();
}



/******************************************************************************
 *
 * PVInspector::PVWorkspacesTabWidget::remove_listing
 *
 *****************************************************************************/
void PVInspector::PVWorkspacesTabWidget::remove_listing(PVTabSplitter* tab)
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
 * PVInspector::PVWorkspacesTabWidget::show_hide_views_tab_widget_Slot
 *
 *****************************************************************************/
void PVInspector::PVWorkspacesTabWidget::show_hide_views_tab_widget_Slot(bool visible)
{
	setVisible(visible);
}



/******************************************************************************
 *
 * PVInspector::PVWorkspacesTabWidget::tabCloseRequested_Slot
 *
 *****************************************************************************/
void PVInspector::PVWorkspacesTabWidget::tabCloseRequested_Slot(int index)
{
	PVTabSplitter* tab = dynamic_cast<PVTabSplitter*>(widget(index));
	assert(tab);
	main_window->close_source(tab);
}
