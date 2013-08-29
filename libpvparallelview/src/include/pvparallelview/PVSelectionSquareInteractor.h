
#ifndef PVPARALLELVIEW_PVSELECTIONSQUAREINTERACTOR_H
#define PVPARALLELVIEW_PVSELECTIONSQUAREINTERACTOR_H

#include <pvkernel/widgets/PVGraphicsView.h>
#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

class QKeyEvent;
class QMouseEvent;

namespace PVParallelView
{

class PVSelectionSquare;

class PVSelectionSquareInteractor: public PVWidgets::PVGraphicsViewInteractor<PVWidgets::PVGraphicsView>
{
public:
	PVSelectionSquareInteractor(PVWidgets::PVGraphicsView* parent, PVSelectionSquare* selection_square);

	bool keyPressEvent(PVWidgets::PVGraphicsView* view, QKeyEvent* event) override;
	bool mousePressEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;
	bool mouseReleaseEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;
	bool mouseMoveEvent(PVWidgets::PVGraphicsView* view, QMouseEvent* event) override;

private:
	PVSelectionSquare *_selection_square;
};

}

#endif // PVPARALLELVIEW_PVSELECTIONSQUAREINTERACTOR_H
