
#ifndef PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H
#define PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVScatterView;

class PVScatterViewInteractor: public PVZoomableDrawingAreaInteractor
{
public:
	PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr);

public:
	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event) override;

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*) override;

protected:
	static PVScatterView *get_scatter_view(PVZoomableDrawingArea *zda);
};

}

#endif // PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H
