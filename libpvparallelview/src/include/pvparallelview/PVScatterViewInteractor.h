/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H
#define PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

namespace PVParallelView
{

class PVScatterView;

class PVScatterViewInteractor : public PVZoomableDrawingAreaInteractor
{
  public:
	explicit PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent = nullptr);

  public:
	bool keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent* event) override;

	bool resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*) override;

  protected:
	static PVScatterView* get_scatter_view(PVZoomableDrawingArea* zda);
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERVIEWINTERACTOR_H
