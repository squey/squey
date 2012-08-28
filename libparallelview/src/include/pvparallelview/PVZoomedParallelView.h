/**
 * \file PVZoomedParallelView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <pvparallelview/PVZoomedParallelScene.h>

#include <QGraphicsView>

namespace PVParallelView
{

class PVZoomedParallelView : public QGraphicsView
{
public:
	PVZoomedParallelView(QWidget *parent = nullptr) :
		QGraphicsView(parent)
	{
	}

	void resizeEvent(QResizeEvent *event)
	{
		PVZoomedParallelScene *zps = (PVZoomedParallelScene*)scene();
		if(zps != nullptr) {
			zps->resize_display(event->size());
		}
		QGraphicsView::resizeEvent(event);
	}
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
