
#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORMAJORY_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORMAJORY_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVZoomableDrawingAreaInteractorMajorY : public PVParallelView::PVZoomableDrawingAreaInteractor
{
public:
	PVZoomableDrawingAreaInteractorMajorY(PVWidgets::PVGraphicsView* parent);

protected:
	bool mousePressEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event) override;

	bool mouseMoveEvent(PVParallelView::PVZoomableDrawingArea* zda, QMouseEvent* event) override;

	bool wheelEvent(PVParallelView::PVZoomableDrawingArea* zda, QWheelEvent* event) override;

private:
	QPoint _pan_reference;
};

}

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORMAJORY_H
