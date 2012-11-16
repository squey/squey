/**
 * \file PVZoomedParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <QScrollBar>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QWidget *parent) :
	QGraphicsView(parent)
{
	setCursor(Qt::CrossCursor);
	setMinimumHeight(300);

	verticalScrollBar()->setObjectName("verticalScrollBar_of_PVListingView");
	horizontalScrollBar()->setObjectName("horizontalScrollBar_of_PVListingView");
}


/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::~PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (scene()) {
		scene()->deleteLater();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::resizeEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent *event)
{
	QGraphicsView::resizeEvent(event);

	PVParallelView::PVZoomedParallelScene *zps = (PVParallelView::PVZoomedParallelScene*)scene();
	if(zps != nullptr) {
		zps->resize_display();
	}
}
