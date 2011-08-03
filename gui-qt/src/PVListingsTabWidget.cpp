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

#ifdef CUSTOMER_RELEASE
	setTabsClosable(true);
	connect(tabBar(), SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested_Slot(int)));
#endif	// CUSTOMER_RELEASE
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
	PVTabSplitter *current_tab = dynamic_cast<PVTabSplitter*>(widget(index));

	if (!current_tab) {
		PVLOG_ERROR("PVInspector::PVListingsTabWidget::%s We have a strange beast in the tab widget!\n", __FUNCTION__);
	}

	removeTab(index);
/*  DDX: I don't understand why this code exists.if (index == 0) {
		currentWidget()->hide();
	}*/

	main_window->destroy_pvgl_views(current_tab->get_lib_view());
	current_tab->deleteLater();

	if(currentIndex() == -1)
	{
		emit is_empty();
		hide();
	}
}

