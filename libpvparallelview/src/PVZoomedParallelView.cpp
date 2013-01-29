/**
 * \file PVZoomedParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <QScrollBar64>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QWidget *parent) :
	PVWidgets::PVGraphicsView(parent)
{
	setCursor(Qt::CrossCursor);
	setMinimumHeight(300);

	get_vertical_scrollbar()->setObjectName("verticalScrollBar_of_PVListingView");
	get_horizontal_scrollbar()->setObjectName("horizontalScrollBar_of_PVListingView");
}


/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::~PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (get_scene()) {
		get_scene()->deleteLater();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent *event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);

	PVParallelView::PVZoomedParallelScene *zps = (PVParallelView::PVZoomedParallelScene*)get_scene();
	if(zps != nullptr) {
		zps->resize_display();
	}
}
