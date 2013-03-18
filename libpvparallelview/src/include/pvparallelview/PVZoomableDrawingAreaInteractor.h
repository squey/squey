#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTOR_H

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>
#include <pvparallelview/PVZoomableDrawingArea.h>

namespace PVParallelView {

typedef PVWidgets::PVGraphicsViewInteractor<PVZoomableDrawingArea> PVZoomableDrawingAreaInteractor;

class PVZoomableDrawingAreaInteractorSameZoom: public PVZoomableDrawingAreaInteractor
{

	friend class PVWidgets::PVGraphicsView;

protected:
	PVZoomableDrawingAreaInteractorSameZoom(PVWidgets::PVGraphicsView* parent);

protected:
	bool wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event) override;
	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent* event) override;
};

}


#endif
