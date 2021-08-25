//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvparallelview/PVHitCountViewInteractor.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountViewParamsWidget.h>

#include <pvkernel/widgets/PVHelpWidget.h>

#include <QScrollBar>

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::PVHitCountViewInteractor
 *****************************************************************************/

PVParallelView::PVHitCountViewInteractor::PVHitCountViewInteractor(
    PVWidgets::PVGraphicsView* parent)
    : PVZoomableDrawingAreaInteractor(parent)
{
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::resizeEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::resizeEvent(PVZoomableDrawingArea* zda,
                                                           QResizeEvent*)
{
	PVHitCountView* hcv = get_hit_count_view(zda);
	hcv->set_x_axis_zoom();

	hcv->reconfigure_view();

	hcv->request_auto_scale();

	hcv->_sel_rect->set_handles_scale(1. / hcv->get_transform().m11(),
	                                  1. / hcv->get_transform().m22());

	if (hcv->get_viewport()) {
		hcv->get_viewport()->update();
	}

	hcv->set_params_widget_position();

	return false;
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::keyPressEvent(PVZoomableDrawingArea* zda,
                                                             QKeyEvent* event)
{
	PVHitCountView* hcv = get_hit_count_view(zda);

	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (hcv->help_widget()->isHidden()) {
			hcv->help_widget()->popup(hcv->get_viewport(),
			                          PVWidgets::PVTextPopupWidget::AlignCenter,
			                          PVWidgets::PVTextPopupWidget::ExpandAll);
			// FIXME : This is a hack to update the help_widget. It should be
			// updated automaticaly as it does with QWebView but it doesn't
			// with QWebEngineView
			hcv->raise();
		}
		return false;
	}

	switch (event->key()) {
	case Qt::Key_Escape:
		hcv->_sel_rect->clear();
		zda->get_viewport()->update();
		break;
	case Qt::Key_Home:
		if (event->modifiers() == Qt::ControlModifier) {
			hcv->set_x_zoom_level_from_sel();

			hcv->reconfigure_view();

			QScrollBar* sb = hcv->get_horizontal_scrollbar();
			sb->setValue(0);

			zda->get_viewport()->update();
			zoom_has_changed(zda, PVZoomableDrawingAreaConstraints::X);
		} else {
			hcv->reset_view();
			hcv->reconfigure_view();
			hcv->_update_all_timer.start();
			hcv->_sel_rect->set_handles_scale(1. / hcv->get_transform().m11(),
			                                  1. / hcv->get_transform().m22());
		}
		return true;
	default:
		break;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::wheelEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::wheelEvent(PVZoomableDrawingArea* zda,
                                                          QWheelEvent* event)
{
	int mask = 0;

	if (event->modifiers() == Qt::NoModifier) {
		mask = PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ControlModifier) {
		mask = PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ShiftModifier) {
		mask = PVZoomableDrawingAreaConstraints::X;
	}

	PVHitCountView* hcv = get_hit_count_view(zda);
	int inc = (event->angleDelta().y() > 0) ? 1 : -1;

	if (mask & PVZoomableDrawingAreaConstraints::X) {

		event->setAccepted(true);

		if (increment_zoom_value(hcv, mask, inc)) {
			QPointF scene_pos = hcv->map_margined_to_scene(QPointF(0, 0));

			hcv->reconfigure_view();

			int scroll_x = hcv->map_to_view(hcv->map_margined_from_scene(scene_pos)).x();
			hcv->get_horizontal_scrollbar()->setValue(scroll_x);

			hcv->get_viewport()->update();
			zoom_has_changed(hcv, mask);
			return true;
		}
	} else if (mask != 0) {
		int inc = (event->angleDelta().y() > 0) ? 1 : -1;

		event->setAccepted(true);

		if (increment_zoom_value(zda, mask, inc)) {
			hcv->request_auto_scale();
			zda->reconfigure_view();
			zda->get_viewport()->update();
			zoom_has_changed(zda, mask);
			return true;
		}
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::get_hit_count_view
 *****************************************************************************/

PVParallelView::PVHitCountView*
PVParallelView::PVHitCountViewInteractor::get_hit_count_view(PVZoomableDrawingArea* zda)
{
	assert(qobject_cast<PVHitCountView*>(zda));
	return static_cast<PVHitCountView*>(zda);
}
