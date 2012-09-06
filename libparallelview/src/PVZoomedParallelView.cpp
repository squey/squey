
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent *event)
{
	QGraphicsView::resizeEvent(event);

	PVParallelView::PVZoomedParallelScene *zps = (PVParallelView::PVZoomedParallelScene*)scene();
	if(zps != nullptr) {
		zps->resize_display();
	}
}
