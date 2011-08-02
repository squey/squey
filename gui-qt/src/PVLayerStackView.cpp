//! \file PVLayerStackView.cpp
//! $Id: PVLayerStackView.cpp 2501 2011-04-25 14:56:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>
#include <QEvent>

#include <pvkernel/core/general.h>
//#include <picviz/PVView.h>

#include <PVMainWindow.h>

#include <PVLayerStackView.h>

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::PVLayerStackView
 *
 *****************************************************************************/
PVInspector::PVLayerStackView::PVLayerStackView(PVMainWindow *mw, PVLayerStackModel *model, PVLayerStackWidget *parent) : QTableView(parent)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	main_window = mw;

	setMinimumSize(0,0);
	setSizePolicy(QSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding));
	setFocusPolicy(Qt::NoFocus);
	verticalHeader()->hide();
	horizontalHeader()->hide();
	//viewport()->setMouseTracking(true);
	//viewport()->setAttribute(Qt::WA_Hover, true);

	layer_stack_delegate = new PVLayerStackDelegate(mw, this);
	setItemDelegate(layer_stack_delegate);

	layer_stack_event_filter = new PVLayerStackEventFilter(mw, this);
	viewport()->installEventFilter(layer_stack_event_filter);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;

	setModel(model);
	resizeColumnsToContents();
	//resizeRowsToContents();
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackView::leaveEvent
 *
 *****************************************************************************/
void PVInspector::PVLayerStackView::leaveEvent(QEvent * /*event*/)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackView::%s\n", __FUNCTION__);

	mouse_hover_layer_index = -1;
	last_mouse_hover_layer_index = -1;
	viewport()->update();
}
