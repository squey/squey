//! \file PVLayerStackEventFilter.cpp
//! $Id: PVLayerStackEventFilter.cpp 2501 2011-04-25 14:56:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtGui>
#include <QEvent>

#include <pvkernel/core/general.h>

#include <PVLayerStackView.h>
#include <PVMainWindow.h>

#include <PVLayerStackEventFilter.h>


/******************************************************************************
 *
 * PVInspector::PVLayerStackEventFilter::PVLayerStackEventFilter
 *
 *****************************************************************************/
PVInspector::PVLayerStackEventFilter::PVLayerStackEventFilter(PVMainWindow *mw, PVLayerStackView *parent) : QObject(parent)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackEventFilter::%s\n", __FUNCTION__);

	main_window = mw;
	layer_stack_view = parent;
}

/******************************************************************************
 *
 * PVInspector::PVLayerStackEventFilter::eventFilter
 *
 *****************************************************************************/
bool PVInspector::PVLayerStackEventFilter::eventFilter(QObject * /*watched*/, QEvent *event)
{
	PVLOG_DEBUG("PVInspector::PVLayerStackEventFilter::%s : with an event of type %d\n", __FUNCTION__, event->type());

	if (event->type() == QEvent::HoverMove) {
		layer_stack_view->mouse_hover_layer_index = layer_stack_view->rowAt(((QMouseEvent *)event)->x());
		if (layer_stack_view->mouse_hover_layer_index != layer_stack_view->last_mouse_hover_layer_index) {
			layer_stack_view->last_mouse_hover_layer_index = layer_stack_view->mouse_hover_layer_index;
			layer_stack_view->viewport()->update();
		}
		return true;
	}
	return false;
}






