
#include <pvparallelview/PVHitCountViewInteractor.h>
#include <pvparallelview/PVHitCountView.h>
#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountViewParamsWidget.h>

#include <pvkernel/widgets/PVHelpWidget.h>

#include <QScrollBar64>

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::PVHitCountViewInteractor
 *****************************************************************************/

PVParallelView::PVHitCountViewInteractor::PVHitCountViewInteractor(PVWidgets::PVGraphicsView* parent) :
	PVZoomableDrawingAreaInteractor(parent)
{}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::resizeEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*)
{
	PVHitCountView *hcv = get_hit_count_view(zda);
	hcv->set_x_axis_zoom();

	hcv->reconfigure_view();

	hcv->_sel_rect->set_handles_scale(1. / hcv->get_transform().m11(),
	                                  1. / hcv->get_transform().m22());

	if (hcv->get_viewport()) {
		hcv->get_viewport()->update();
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::keyPressEvent(PVZoomableDrawingArea* zda, QKeyEvent *event)
{
	PVHitCountView *hcv = get_hit_count_view(zda);

	if(PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (hcv->help_widget()->isHidden()) {
			hcv->help_widget()->popup(hcv->get_viewport(),
			                          PVWidgets::PVTextPopupWidget::AlignCenter,
			                          PVWidgets::PVTextPopupWidget::ExpandAll, 16);
		}
		return false;
	}

	switch (event->key()) {
	case Qt::Key_Escape:
		hcv->_sel_rect->clear();
		zda->get_viewport()->update();
		break;
	case Qt::Key_Space:
		if (event->modifiers() == Qt::NoModifier) {
			if(hcv->params_widget()->isHidden()) {
				hcv->params_widget()->setPersistence(false);
				hcv->params_widget()->popup(QCursor::pos(), true);
				return true;
			}
		} else if (event->modifiers() == Qt::ControlModifier) {
			hcv->params_widget()->setPersistence(true);
			hcv->params_widget()->popup(QCursor::pos(), true);
			return true;
		}
		break;
	case Qt::Key_Home:
		if (event->modifiers() == Qt::ControlModifier) {
			hcv->set_x_zoom_level_from_sel();

			hcv->reconfigure_view();

			QScrollBar64 *sb = hcv->get_horizontal_scrollbar();
			sb->setValue(0);

			zda->get_viewport()->update();
			zoom_has_changed(zda, PVZoomableDrawingAreaConstraints::X);
		}
		else {
			hcv->reset_view();
			hcv->reconfigure_view();
			hcv->_update_all_timer.start();
			hcv->_sel_rect->set_handles_scale(1. / hcv->get_transform().m11(),
			                                  1. / hcv->get_transform().m22());
		}
		return true;
	case Qt::Key_S:
		if (event->modifiers() == Qt::AltModifier) {
			hcv->toggle_auto_x_zoom_sel();
			return true;
		}
		break;
	default:
		break;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewInteractor::wheelEvent
 *****************************************************************************/

bool PVParallelView::PVHitCountViewInteractor::wheelEvent(PVZoomableDrawingArea* zda, QWheelEvent* event)
{
	int mask = 0;

	if (event->modifiers() == Qt::NoModifier) {
		mask = PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ControlModifier) {
		mask = PVZoomableDrawingAreaConstraints::X | PVZoomableDrawingAreaConstraints::Y;
	} else if (event->modifiers() == Qt::ShiftModifier) {
		mask = PVZoomableDrawingAreaConstraints::X;
	}

	PVHitCountView *hcv = get_hit_count_view(zda);
	int inc = (event->delta() > 0)?1:-1;

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
	} else 	if (mask != 0) {
		int inc = (event->delta() > 0)?1:-1;

		event->setAccepted(true);

		if (increment_zoom_value(zda, mask, inc)) {
			hcv->_do_auto_scale = hcv->_auto_x_zoom_sel;
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
PVParallelView::PVHitCountViewInteractor::get_hit_count_view(PVZoomableDrawingArea *zda)
{
	assert(qobject_cast<PVHitCountView*>(zda));
	return static_cast<PVHitCountView*>(zda);
}
