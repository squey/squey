/**
 * \file PVFieldSplitterIronPortMailParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2013
 */

#include "PVFieldSplitterIronPortMailParamWidget.h"
#include "PVFieldSplitterIronPortMail.h"

#include <QAction>

/******************************************************************************
 * PVFilter::PVFieldSplitterIronPortMailParamWidget::PVFieldSplitterIronPortMailParamWidget
 *****************************************************************************/

PVFilter::PVFieldSplitterIronPortMailParamWidget::PVFieldSplitterIronPortMailParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterIronPortMail()))
{
	PVLOG_DEBUG("constructor PVFieldSplitterIronPortMailParamWidget\n");
	_menu_action = new QAction(QString("add IronPort Mail Splitter"),
	                           nullptr);
}

/******************************************************************************
 * PVFilter::PVFieldSplitterIronPortMailParamWidget::get_action_menu
 *****************************************************************************/

QAction* PVFilter::PVFieldSplitterIronPortMailParamWidget::get_action_menu()
{
	assert(_menu_action);
	return _menu_action;
}

/******************************************************************************
 * PVFilter::PVFieldSplitterIronPortMailParamWidget::get_param_widget
 *****************************************************************************/

QWidget* PVFilter::PVFieldSplitterIronPortMailParamWidget::get_param_widget()
{
	/* to correctly initialize the fields tree when a new splitter is
	 * added, the 2 following things have to be done
	 */
	set_child_count(PVFieldSplitterIronPortMail::COL_FIELDS_NUMBER);
	emit nchilds_changed_Signal();

	// no need for a param widget
	return nullptr;
}
