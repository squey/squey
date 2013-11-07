
#ifndef PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H
#define PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H

#include <pvkernel/widgets/PVGraphicsViewInteractor.h>

namespace PVWidgets
{

class PVGraphicsViewInteractorScene : public PVGraphicsViewInteractor<PVGraphicsView>
{
public:
	PVGraphicsViewInteractorScene(PVGraphicsView* parent);

protected:
	bool contextMenuEvent(PVGraphicsView* obj, QContextMenuEvent* event) override;

	bool mouseDoubleClickEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mousePressEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mouseReleaseEvent(PVGraphicsView* obj, QMouseEvent* event) override;
	bool mouseMoveEvent(PVGraphicsView* obj, QMouseEvent* event) override;

	bool wheelEvent(PVGraphicsView* obj, QWheelEvent* event) override;

	bool keyPressEvent(PVGraphicsView* obj, QKeyEvent* event) override;
	bool keyReleaseEvent(PVGraphicsView* obj, QKeyEvent* event) override;
};

}

#endif // PVWIDGETS_PVGRAPHICSVIEWINTERACTORSCENE_H
