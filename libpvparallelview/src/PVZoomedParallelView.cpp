
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

/*****************************************************************************
 * PVParallelView::PVZoomedParallelView::PVZoomedParallelView
 *****************************************************************************/

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(QWidget *parent) :
	QGraphicsView(parent)
{
	setMinimumHeight(300);
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
