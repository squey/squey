//! \file PVViewsListingView.cpp
//! $Id: PVViewsListingView.cpp 2501 2011-04-25 14:56:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2012
//! Copyright (C) Philippe Saadé 2009-2012
//! Copyright (C) Picviz Labs 2012


#include <PVViewsListingView.h>
#include <PVViewsModel.h>
#include <PVTabSplitter.h>

/******************************************************************************
 *
 * PVInspector::PVViewsListingView::PVViewsListingView
 *
 *****************************************************************************/
PVInspector::PVViewsListingView::PVViewsListingView(PVViewsModel* model, PVTabSplitter* tab, QWidget* parent):
	QTreeView(parent),
	_tab(tab),
	_model(model)
{
	// SIZE STUFF
	setMinimumSize(100,0);
	setSizePolicy(QSizePolicy::Minimum,QSizePolicy::Expanding);

	setHeaderHidden(true);
	setModel(model);
	setAllColumnsShowFocus(true);
}



/******************************************************************************
 *
 * PVInspector::PVViewsListingView::mouseDoubleClickEvent
 *
 *****************************************************************************/
void PVInspector::PVViewsListingView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);

	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		PVLOG_INFO("index not valid\n");
		return;
	}

	PVViewsModel::PVIndexNode node_obj(_model->get_object(idx_click));
	if (node_obj.is_plotted()) {
		_tab->select_plotted(node_obj.as_plotted());
		_model->emitDataChanged(idx_click);
	}
}
