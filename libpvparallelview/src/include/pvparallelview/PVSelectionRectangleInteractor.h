
#ifndef PVPARALLELVIEW_PVSELECTIONRECTANGLEINTERACTOR_H
#define PVPARALLELVIEW_PVSELECTIONRECTANGLEINTERACTOR_H

#include <pvkernel/widgets/PVGraphicsView.h>
#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

class QKeyEvent;
class QMouseEvent;

namespace PVParallelView
{

class PVSelectionRectangle;

class PVSelectionRectangleInteractor: public PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>
{
public:
	PVSelectionRectangleInteractor(PVWidgets::PVGraphicsView* parent,
	                               PVSelectionRectangle* selection_rectangle);

	bool keyPressEvent(PVWidgets::PVGraphicsView* view, QKeyEvent* event) override;
	bool mousePressEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;
	bool mouseReleaseEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;
	bool mouseMoveEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;

private:
	PVParallelView::PVSelectionRectangle *_selection_rectangle;
};

}

#endif // PVPARALLELVIEW_PVSELECTIONRECTANGLEINTERACTOR_H
