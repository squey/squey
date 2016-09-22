/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORHOMOTHETIC_H
#define PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORHOMOTHETIC_H

#include <pvparallelview/PVZoomableDrawingAreaInteractor.h>

#include <QPoint>

namespace PVParallelView
{

class PVZoomableDrawingAreaInteractorHomothetic : public PVZoomableDrawingAreaInteractor
{
  public:
	explicit PVZoomableDrawingAreaInteractorHomothetic(PVWidgets::PVGraphicsView* parent);

  protected:
	bool mousePressEvent(PVZoomableDrawingArea* zda, QMouseEvent* event) override;

	bool mouseMoveEvent(PVZoomableDrawingArea* zda, QMouseEvent* event) override;

	bool wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event) override;

  private:
	QPoint _pan_reference;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMABLEDRAWINGAREAINTERACTORHOMOTHETIC_H
