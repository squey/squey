/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVHitCountView;

class PVHitCountViewInteractor : public PVZoomableDrawingAreaInteractor
{
public:
	PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr);

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*) override;

	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event) override;

	bool wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event) override;

protected:
	static PVHitCountView *get_hit_count_view(PVZoomableDrawingArea *zda);
};

}

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWINTERACTOR_H
