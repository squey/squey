
#include <pvparallelview/PVScatterViewInteractor.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterViewParamsWidget.h>

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::PVScatterViewInteractor
 *****************************************************************************/

PVParallelView::PVScatterViewInteractor::PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent) :
PVZoomableDrawingAreaInteractor(parent)
{
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVScatterViewInteractor::keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event)
{
	PVScatterView *sv = get_scatter_view(zda);
	switch (event->key()) {
	case Qt::Key_Escape:
		sv->_sel_rect->clear();
		sv->get_viewport()->update();
		break;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::resizeEvent
 *****************************************************************************/

bool PVParallelView::PVScatterViewInteractor::resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*)
{
	PVScatterView *sv = get_scatter_view(zda);

	sv->do_update_all();

	sv->reconfigure_view();

	sv->_sel_rect->set_handles_scale(1. / sv->get_transform().m11(),
	                                 1. / sv->get_transform().m22());

	if (sv->get_viewport()) {
		sv->get_viewport()->update();
	}

	sv->set_params_widget_position();

	return false;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::get_scatter_view
 *****************************************************************************/

PVParallelView::PVScatterView*
PVParallelView::PVScatterViewInteractor::get_scatter_view(PVZoomableDrawingArea *zda)
{
	assert(qobject_cast<PVScatterView*>(zda));
	return static_cast<PVScatterView*>(zda);
}
