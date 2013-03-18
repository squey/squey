#include <pvkernel/core/general.h>
#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

PVParallelView::PVZoomableDrawingAreaInteractorSameZoom::PVZoomableDrawingAreaInteractorSameZoom(PVWidgets::PVGraphicsView* parent):
	PVZoomableDrawingAreaInteractor(parent)
{
}

bool PVParallelView::PVZoomableDrawingAreaInteractorSameZoom::wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	PVLOG_INFO("In wheelEvent interactor\n");
	return false;
}

bool PVParallelView::PVZoomableDrawingAreaInteractorSameZoom::keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent* event)
{
	PVLOG_INFO("In keyPressEvent interactor\n");
	return false;
}
