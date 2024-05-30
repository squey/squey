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

#include <pvkernel/core/PVAlgorithms.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/qmetaobject_helper.h>

#include <pvkernel/widgets/PVHelpWidget.h>

#include <squey/PVSelection.h>

#include <pvparallelview/PVAbstractAxisSlider.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoomedSelectionAxisSliders.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>
#include <pvparallelview/PVZoomedParallelViewSelectionLine.h>

#include <QMetaObject>
#include <QThread>
#include <QScrollBar>
#include <QPainter>

/**
 * IDEA: keep ratio between the width of the 2 zones in full parallel view
 *       => stores the scale for each beta (for the left one and the right one)
 * RH: AG said this feature is not still planned (TODO -> IDEA)
 */

/* NOTE: when zooming, the smallest backend_image's height is 1024 (2048 / 2).
 *       So with a view's greater than 1024, there will be an empty space at
 *       the view's bottom. So the view's height must be limited to 1024. And
 *       limiting its width to 1024 is not so weird.
 *       The QGraphicsPixmapItems height has to be limited to 1024 too: the
 *       restriction for BCI codes is only on the zoomed axis, not on their
 *       neighbours. The QGraphicsPixmapItems could also contains a trapezoid
 *       containing lines, which is ugly. Having a height of 1024 remove this
 *       problem.
 */

#define ZOOM_MODIFIER Qt::NoModifier
#define PAN_MODIFIER Qt::ControlModifier
#define SLOW_PAN_MODIFIER Qt::ShiftModifier

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene(
    PVParallelView::PVZoomedParallelView* zpview,
    Squey::PVView& pvview_sp,
    PVParallelView::PVSlidersManager* sliders_manager_p,
    PVZonesProcessor& zp_sel,
    PVZonesProcessor& zp_bg,
    PVZonesManager const& zm,
    PVCombCol axis_index)
    : QGraphicsScene(zpview)
    , _zpview(zpview)
    , _pvview(pvview_sp)
    , _sliders_manager_p(sliders_manager_p)
    , _axis_index(axis_index)
    , _nraw_col(_pvview.get_axes_combination().get_nraw_axis(_axis_index))
    , _zm(zm)
    , _pending_deletion(false)
    , _left_zone(nullptr)
    , _right_zone(nullptr)
    , _show_bg(true)
    , _zp_sel(zp_sel)
    , _zp_bg(zp_bg)
    , _selection_sliders(nullptr)
    , _view_deleted(false)
{
	_zpview->set_viewport_cursor(Qt::CrossCursor);

	setItemIndexMethod(QGraphicsScene::NoIndex);

	_zpview->set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
	_zpview->set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);
	_zpview->set_resize_anchor(PVWidgets::PVGraphicsView::AnchorViewCenter);
	_zpview->set_transformation_anchor(PVWidgets::PVGraphicsView::AnchorUnderMouse);

	_zpview->setMaximumWidth(1024);
	_zpview->setMaximumHeight(1024);

	_sel_line = new PVZoomedParallelViewSelectionLine(_zpview);
	_sel_line->setZValue(1.e43);

	addItem(_sel_line);
	connect(_sel_line, &PVZoomedParallelViewSelectionLine::commit_volatile_selection, this,
	        &PVZoomedParallelScene::commit_volatile_selection_Slot);

	setSceneRect(-512, 0, 1024, 1024);

	connect(_zpview->get_vertical_scrollbar(), &QScrollBar::valueChanged, this,
	        &PVZoomedParallelScene::scrollbar_changed_Slot);

	connect(_zpview->params_widget(), &PVZoomedParallelViewParamsWidget::change_to_col, this,
	        &PVZoomedParallelScene::change_to_col);
	
	_zpview->update_window_title(_pvview, axis_index);

	_sliders_group =
	    std::make_unique<PVParallelView::PVSlidersGroup>(_sliders_manager_p, _axis_index);
	_sliders_group->setPos(0., 0.);
	_sliders_group->add_zoom_sliders(0, 1024);

	// the sliders must be over all other QGraphicsItems
	_sliders_group->setZValue(1.e42);

	addItem(_sliders_group.get());

	configure_axis(true);

	// Register view for unselected & zombie events toggle
	pvview_sp._toggle_unselected_zombie_visibility.connect(sigc::mem_fun(
	    *this, &PVParallelView::PVZoomedParallelScene::toggle_unselected_zombie_visibility));

	sliders_manager_p->_update_zoom_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVZoomedParallelScene::on_zoom_sliders_update));
	sliders_manager_p->_del_zoom_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVZoomedParallelScene::on_zoom_sliders_del));
	sliders_manager_p->_del_zoomed_selection_sliders.connect(
	    sigc::mem_fun(*this, &PVParallelView::PVZoomedParallelScene::on_zoomed_sel_sliders_del));

	_updateall_timer.setInterval(150);
	_updateall_timer.setSingleShot(true);
	connect(&_updateall_timer, &QTimer::timeout, this,
	        &PVZoomedParallelScene::updateall_timeout_Slot);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene()
{
	if (_zpview != nullptr) {
		_zpview->set_scene(nullptr);
		_zpview = nullptr;
	}

	if (!_view_deleted) {
		common::get_lib_view(_pvview)->remove_zoomed_view(this);
	}

	if (_selection_sliders) {
		_selection_sliders->remove_from_axis();
		_selection_sliders = nullptr;
	}

	if (_pending_deletion == false) {
		_pending_deletion = true;
		_sliders_manager_p->del_zoom_sliders(_axis_index, _sliders_group.get());
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	// do item's hover stuff
	QGraphicsScene::mousePressEvent(event);

	if (event->isAccepted()) {
		// a QGraphicsItem has already done something (usually a contextMenuEvent)
		return;
	}

	if (event->button() == Qt::RightButton) {
		_pan_reference_y = event->screenPos().y();
		event->accept();
	} else if (!_sliders_group->sliders_moving() && (event->button() == Qt::LeftButton)) {
		_sel_line->begin(event->scenePos());
		if (_selection_sliders) {
			_selection_sliders->hide();
		}
		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	if (!_sliders_group->sliders_moving() && (event->button() == Qt::LeftButton)) {
		if (_sel_line->is_null()) {
			_sel_line->clear();
			if (_selection_sliders) {
				_selection_sliders->remove_from_axis();
				_selection_sliders = nullptr;
			}
		} else {
			_sel_line->end(event->scenePos());
			int64_t y_min = _sel_line->top() * BUCKET_ELT_COUNT;
			int64_t y_max = _sel_line->bottom() * BUCKET_ELT_COUNT;

			if (_selection_sliders == nullptr) {
				_selection_sliders = _sliders_group->add_zoomed_selection_sliders(y_min, y_max);
			}
			/* ::set_value implictly emits the signal slider_moved;
			 * what will lead to a selection update in PVFullParallelView
			 */
			_selection_sliders->set_value(y_min, y_max);
			_selection_sliders->show();
		}

		event->accept();
	}

	// do item's hover stuff
	QGraphicsScene::mouseReleaseEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	if (event->buttons() == Qt::RightButton) {
		QScrollBar* sb = _zpview->get_vertical_scrollbar();
		qint64 delta = _pan_reference_y - event->screenPos().y();
		_pan_reference_y = event->screenPos().y();
		qint64 v = sb->value();
		sb->setValue(v + delta);
		event->accept();
	} else if (!_sliders_group->sliders_moving() && (event->buttons() == Qt::LeftButton)) {
		_sel_line->step(event->scenePos());
		event->accept();
	}

	// do item's hover stuff
	QGraphicsScene::mouseMoveEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::wheelEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	if (event->modifiers() == ZOOM_MODIFIER) {
		// zoom
		if (event->delta() > 0) {
			if (_wheel_value < max_wheel_value) {
				++_wheel_value;
				update_zoom();
			}
		} else {
			if (_wheel_value > 0) {
				--_wheel_value;
				update_zoom();
			}
		}
	} else if (event->modifiers() == SLOW_PAN_MODIFIER) {
		// precise panning
		QScrollBar* sb = _zpview->get_vertical_scrollbar();
		if (event->delta() > 0) {
			qint64 v = sb->value();
			if (v > sb->minimum()) {
				sb->setValue(v - 1);
			}
		} else {
			qint64 v = sb->value();
			if (v < sb->maximum()) {
				sb->setValue(v + 1);
			}
		}
	} else if (event->modifiers() == PAN_MODIFIER) {
		// panning
		QScrollBar* sb = _zpview->get_vertical_scrollbar();
		if (event->delta() > 0) {
			sb->triggerAction(QAbstractSlider::SliderSingleStepSub);
		} else {
			sb->triggerAction(QAbstractSlider::SliderSingleStepAdd);
		}
	}

	event->accept();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::keyPressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::keyPressEvent(QKeyEvent* event)
{
	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (_zpview->help_widget()->isHidden()) {
			_zpview->help_widget()->popup(_zpview->get_viewport(),
			                              PVWidgets::PVTextPopupWidget::AlignCenter,
			                              PVWidgets::PVTextPopupWidget::ExpandAll);
			// FIXME : This is a hack to update the help_widget. It should be
			// updated automaticaly as it does with QWebView but it doesn't
			// with QWebEngineView
			_zpview->raise();
			event->accept();
		}
		return;
	}

	if (event->key() == Qt::Key_Escape) {
		_sel_line->clear();
		if (_selection_sliders) {
			_selection_sliders->remove_from_axis();
			_selection_sliders = nullptr;
		}
		event->accept();
	}
#ifdef SQUEY_DEVELOPER_MODE
	else if (event->key() == Qt::Key_Space) {
		PVLOG_INFO("PVZoomedParallelScene: forcing full redraw\n");
		update_all();
		event->accept();
	}
#endif
}

void PVParallelView::PVZoomedParallelScene::update(const QRectF& rect)
{
	QGraphicsScene::update(rect);
	if (_zpview) {
		_zpview->get_viewport()->update();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zones
 *****************************************************************************/

bool PVParallelView::PVZoomedParallelScene::update_zones()
{
	PVCombCol axis = _pvview.get_axes_combination().get_first_comb_col(_nraw_col);

	if (axis == PVCombCol()) {
		/* a candidate can not be found to replace the old
		 * axis; the zoom view must be closed.
		 */
		return false;
	}

	/* the axes has only been moved, nothing special to do.
	*/
	_axis_index = axis;

	configure_axis();

	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::change_to_col
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::change_to_col(PVCombCol index)
{
	_axis_index = index;
	_nraw_col = _pvview.get_axes_combination().get_nraw_axis(index);

	if (_selection_sliders) {
		_selection_sliders->remove_from_axis();
		_selection_sliders = nullptr;
	}

	removeItem(_sliders_group.get());
	_sliders_group->delete_own_zoom_slider();

	_sliders_group =
	    std::make_unique<PVParallelView::PVSlidersGroup>(_sliders_manager_p, _axis_index);
	_sliders_group->setPos(0., 0.);
	_sliders_group->add_zoom_sliders(0, 1024);

	// the sliders must be over all other QGraphicsItems
	_sliders_group->setZValue(1.e42);

	addItem(_sliders_group.get());

	configure_axis(true);

	update_all();

	_zpview->update_window_title(_pvview, index);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::configure_axis
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::configure_axis(bool reset_view_param)
{
	if (reset_view_param) {
		/* reset zoom
		 */
		_wheel_value = 0;
		update_zoom();
		/* and pan. The scrollbar must be reset after after the zoom
		 * update,otherwise, zooming do a pan (really strange...)
		 */
		_zpview->get_vertical_scrollbar()->setValue(0);
	}

	/* have a coherent param widget
	 */
	_zpview->params_widget()->build_axis_menu(_axis_index);

	if (_axis_index == PVCombCol()) {
		return;
	}

	/* get the needed zones
	 */
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(false);
		    common::get_lib_view(_pvview)->request_zoomed_zone_trees(_axis_index);
		},
	    "Initializing zoomed parallel view...", nullptr);

	/* the zones
	 */
	// needed pixmap to create QGraphicsPixmapItem
	QPixmap dummy_pixmap;

	if (_left_zone) {
		removeItem(_left_zone->item);
		delete _left_zone->item;
		_left_zone.reset();
	}

	if (_right_zone) {
		removeItem(_right_zone->item);
		delete _right_zone->item;
		_right_zone.reset();
	}

	_renderable_zone_number = 0;

	if (_axis_index > 0) {
		_left_zone = std::make_unique<zone_desc_t>();
		_left_zone->bg_image = common::backend().create_image(image_width, bbits);
		_left_zone->sel_image = common::backend().create_image(image_width, bbits);
		_left_zone->item = addPixmap(dummy_pixmap);

		++_renderable_zone_number;
	}

	if (size_t(_axis_index) < get_zones_manager().get_number_of_axes_comb_zones()) {
		_right_zone = std::make_unique<zone_desc_t>();
		_right_zone->bg_image = common::backend().create_image(image_width, bbits);
		_right_zone->sel_image = common::backend().create_image(image_width, bbits);
		_right_zone->item = addPixmap(dummy_pixmap);

		++_renderable_zone_number;
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::drawBackground
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::drawBackground(QPainter* painter,
                                                           const QRectF& /*rect*/)
{
	if (_axis_index == PVCombCol()) {
		return;
	}

	QRect screen_rect = _zpview->get_viewport()->rect();
	int screen_center = screen_rect.center().x() + 1;

	// save the painter's state to restore it later because the scene
	// transformation matrix is unneeded for what we have to draw
	QTransform t = painter->transform();
	painter->resetTransform();
	// the pen has to be saved too
	QPen old_pen = painter->pen();

	painter->fillRect(screen_rect, color_view_bg);

	// draw axis
	painter->setPen(
	    QPen(_pvview.get_axis(_axis_index).get_color().toQColor(), PARALLELVIEW_AXIS_WIDTH));
	painter->drawLine(screen_center, 0, screen_center, screen_rect.height());

	// get back the painter's original state
	painter->setTransform(t);

#if 0
	// really usefull to see quadtrees
	painter->setPen(Qt::blue);
	for(int i = 0; i < 1025; ++i) {
		painter->drawLine(QPointF(-10, i), QPointF(10, i));
	}
#endif

	painter->setPen(old_pen);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_display()
{
	_zpview->set_viewport_cursor(Qt::BusyCursor);

	double alpha = bbits_alpha_scale * pow(root_step, get_zoom_step());
	double beta = 1. / get_scale_factor();

	QRectF screen_rect = _zpview->get_viewport()->rect();
	qreal screen_center = screen_rect.center().x();

	QRectF screen_rect_s = _zpview->map_to_scene(screen_rect);
	QRectF view_rect = sceneRect().intersected(screen_rect_s);

	double pixel_height = (1UL << (32 - NBITS_INDEX)) / get_scale_factor();

	// the screen's upper limit in scaled coordinates system
	uint64_t y_min = view_rect.top() * BUCKET_ELT_COUNT;
	// the backend_image's lower limit in scaled coordinates system
	auto y_lim =
	    PVCore::clamp<uint64_t>(y_min + (1 << bbits) * alpha * pixel_height, 0ULL, 1ULL << 32);
	// the screen's lower limit in scaled coordinates system
	// y_max can not be greater than y_lim
	auto y_max =
	    PVCore::clamp<uint64_t>(y_min + screen_rect.height() * pixel_height, 0ULL, y_lim);

	_sliders_manager_p->update_zoom_sliders(_axis_index, _sliders_group.get(), y_min, y_max,
	                                        PVParallelView::PVSlidersManager::ZoomSliderNone);
	_last_y_min = y_min;
	_last_y_max = y_max;

	_next_beta = beta;

	qint64 gap_y = (screen_rect_s.top() < 0) ? round(-screen_rect_s.top()) : 0;

	_renderable_zone_number = 0;
	if (_left_zone) {
		if (_render_type == RENDER_ALL) {
			_left_zone->cancel_last_bg();
		} else {
			// If the background is rendering, we need to wait for one more rendering!
			_renderable_zone_number += (bool)_left_zone->last_zr_bg;
		}
		_left_zone->cancel_last_sel();

		QPointF npos(screen_center - image_width - axis_half_width, gap_y);

		_left_zone->next_pos = _zpview->map_to_scene(npos);
	}

	if (_right_zone) {
		if (_render_type == RENDER_ALL) {
			_right_zone->cancel_last_bg();
		} else {
			// If the background is rendering, we need to wait for one more rendering!
			_renderable_zone_number += (bool)_right_zone->last_zr_bg;
		}
		_right_zone->cancel_last_sel();

		QPointF npos(screen_center + axis_half_width + 1, gap_y);

		_right_zone->next_pos = _zpview->map_to_scene(npos);
	}

	int zoom_level = get_zoom_level();

	if (_left_zone) {
		if (_render_type == RENDER_ALL) {
			_renderable_zone_number++;
			PVZoneRenderingBCI_p<bbits> zr(new PVZoneRenderingBCI<bbits>(
			    left_zone_id(),
			    [&, y_min, y_max, y_lim, zoom_level, beta](
			        PVZoneID const z, PVCore::PVHSVColor const* colors, PVBCICode<bbits>* codes) {
				    zzt_context_t ctxt;
				    return get_zztree(z).browse_bci_bg_by_y2(
				        ctxt, y_min, y_max, y_lim, layer_stack_output_selection(), zoom_level,
				        image_width, colors, codes, beta);
				},
			    _left_zone->bg_image,
			    0, // x_start
			    image_width,
			    alpha,  // zoom_y
			    true)); // reversed

			connect_zr(zr.get(), "zr_finished");
			_left_zone->last_zr_bg = zr;
			_zp_bg.add_job(zr);
		}

		_renderable_zone_number++;
		PVZoneRenderingBCI_p<bbits> zr(new PVZoneRenderingBCI<bbits>(
		    left_zone_id(),
		    [&, y_min, y_max, y_lim, zoom_level,
		     beta](PVZoneID const z, PVCore::PVHSVColor const* colors, PVBCICode<bbits>* codes) {
			    zzt_context_t ctxt;
			    return get_zztree(z).browse_bci_sel_by_y2(ctxt, y_min, y_max, y_lim,
			                                              real_selection(), zoom_level, image_width,
			                                              colors, codes, beta);
			},
		    _left_zone->sel_image,
		    0, // x_start
		    image_width,
		    alpha,  // zoom_y
		    true)); // reversed

		connect_zr(zr.get(), "zr_finished");
		_left_zone->last_zr_sel = zr;
		_zp_sel.add_job(zr);
	}

	if (_right_zone) {
		if (_render_type == RENDER_ALL) {
			_renderable_zone_number++;
			PVZoneRenderingBCI_p<bbits> zr(new PVZoneRenderingBCI<bbits>(
			    right_zone_id(),
			    [&, y_min, y_max, y_lim, zoom_level, beta](
			        PVZoneID const z, PVCore::PVHSVColor const* colors, PVBCICode<bbits>* codes) {
				    zzt_context_t ctxt;
				    return get_zztree(z).browse_bci_bg_by_y1(
				        ctxt, y_min, y_max, y_lim, layer_stack_output_selection(), zoom_level,
				        image_width, colors, codes, beta);
				},
			    _right_zone->bg_image,
			    0, // x_start
			    image_width,
			    alpha,   // zoom_y
			    false)); // reversed

			connect_zr(zr.get(), "zr_finished");
			_right_zone->last_zr_bg = zr;
			_zp_bg.add_job(zr);
		}

		_renderable_zone_number++;
		PVZoneRenderingBCI_p<bbits> zr(new PVZoneRenderingBCI<bbits>(
		    right_zone_id(),
		    [&, y_min, y_max, y_lim, zoom_level,
		     beta](PVZoneID const z, PVCore::PVHSVColor const* colors, PVBCICode<bbits>* codes) {
			    zzt_context_t ctxt;
			    return get_zztree(z).browse_bci_sel_by_y1(ctxt, y_min, y_max, y_lim,
			                                              real_selection(), zoom_level, image_width,
			                                              colors, codes, beta);
			},
		    _right_zone->sel_image,
		    0, // x_start
		    image_width,
		    alpha,   // zoom_y
		    false)); // reversed

		connect_zr(zr.get(), "zr_finished");
		_right_zone->last_zr_sel = zr;
		_zp_sel.add_job(zr);
	}
}

void PVParallelView::PVZoomedParallelScene::connect_zr(PVZoneRenderingBCI<bbits>* zr,
                                                       const char* slot)
{
	zr->set_render_finished_slot(this, slot);
}

void PVParallelView::PVZoomedParallelScene::zr_finished(PVZoneRendering_p zr, PVZoneID zone_id)
{
	assert(QThread::currentThread() == this->thread());
	bool zr_catch = true;

	if (zone_id == left_zone_id()) {
		if (_left_zone->last_zr_sel == zr) {
			_left_zone->last_zr_sel.reset();
		} else if (_left_zone->last_zr_bg == zr) {
			_left_zone->last_zr_bg.reset();
		} else {
			zr_catch = false;
		}
	} else {
		if (_right_zone->last_zr_sel == zr) {
			_right_zone->last_zr_sel.reset();
		} else if (_right_zone->last_zr_bg == zr) {
			_right_zone->last_zr_bg.reset();
		} else {
			zr_catch = false;
		}
	}

	if (!zr_catch || zr->should_cancel()) {
		// Cancellation may have occured between the event posted for this call
		// in the Qt's main loop event and the actual call.
		return;
	}

	_renderable_zone_number--;
	// "<=" just in case something went wrong, but it should'nt..
	if (_renderable_zone_number <= 0) {
		all_rendering_done();
	}
}

/******************************************************************************
 * PVParallelView::PVZoomedParallelScene::toggle_unselected_zombie_visibility
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::toggle_unselected_zombie_visibility()
{
	_show_bg = _pvview.are_view_unselected_zombie_visible();

	recreate_images();

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_zoom(bool need_recomputation)
{
	double scale_factor = get_scale_factor();

	QTransform transfo;
	transfo.scale(scale_factor, scale_factor);
	_zpview->set_transform(transfo);

	_sel_line->set_view_scale(scale_factor, scale_factor);

	/* make sure the scene is always horizontally centered. And because
	 * of the selection rectangle, the scene's bounding box can change;
	 * so that its center could not be 0...
	 */
	QScrollBar* sb = _zpview->get_horizontal_scrollbar();
	int64_t mid = ((int64_t)sb->maximum() + sb->minimum()) / 2;
	sb->setValue(mid);

	qreal center_x = _zpview->get_viewport()->rect().center().x();

	/* now, it's time to tell each QGraphicsPixmapItem where it must be.
	*/
	if (_left_zone) {
		QPointF p = _left_zone->item->pos();
		double screen_left_x = center_x + 1;

		/* the image's size depends on its last computed beta factor
		 * and the current scale factor (it's obvious to prove but I
		 * have no time to:)
		 */
		screen_left_x -= image_width * (_current_beta * scale_factor);
		screen_left_x -= axis_half_width;

		QPointF np = _zpview->map_to_scene(QPointF(screen_left_x, 0));

		_left_zone->item->setPos(np.x(), p.y());
	}

	if (_right_zone) {
		QPointF p = _right_zone->item->pos();
		int screen_right_x = center_x + axis_half_width + 2;

		QPointF np = _zpview->map_to_scene(QPointF(screen_right_x, 0));

		_right_zone->item->setPos(np.x(), p.y());
	}

	if (need_recomputation) {
		_updateall_timer.start();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot(qint64 /*value*/)
{
	_updateall_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::updateall_timeout_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::updateall_timeout_Slot()
{
	update_all();
}

void PVParallelView::PVZoomedParallelScene::update_all_async()
{
	// QMetaObject::invokeMethod(this, &PVZoomedParallelScene::update_all, Qt::QueuedConnection);
	PVCore::invokeMethod(this, &PVZoomedParallelScene::update_all, Qt::QueuedConnection);
}

void PVParallelView::PVZoomedParallelScene::update_new_selection_async()
{
	// QMetaObject::invokeMethod(this, &PVZoomedParallelScene::update_sel, Qt::QueuedConnection);
	PVCore::invokeMethod(this, &PVZoomedParallelScene::update_sel, Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::recreate_images
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::recreate_images()
{
	QImage image(image_width, image_height, QImage::Format_ARGB32);
	QPainter painter(&image);

	if (_left_zone) {
		image.fill(Qt::transparent);

		if (show_bg()) {
			painter.setOpacity(0.25);
			painter.drawImage(0, 0, _left_zone->bg_image->qimage(qimage_height()));
		}
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _left_zone->sel_image->qimage(qimage_height()));

		_left_zone->item->setPixmap(QPixmap::fromImage(image));
	}

	if (_right_zone) {
		image.fill(Qt::transparent);

		if (show_bg()) {
			painter.setOpacity(0.25);
			painter.drawImage(0, 0, _right_zone->bg_image->qimage(qimage_height()));
		}
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _right_zone->sel_image->qimage(qimage_height()));

		_right_zone->item->setPixmap(QPixmap::fromImage(image));
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::all_rendering_done
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::all_rendering_done()
{
	_current_beta = _next_beta;

	recreate_images();

	if (_left_zone) {
		_left_zone->item->setPos(_left_zone->next_pos);
		_left_zone->item->setScale(_current_beta);
	}

	if (_right_zone) {
		_right_zone->item->setPos(_right_zone->next_pos);
		_right_zone->item->setScale(_current_beta);
	}

	update();
	_zpview->set_viewport_cursor(Qt::CrossCursor);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot()
{
	if (_sel_line->is_null()) {
		// force selection update
		_pvview.select_none();
	} else {
		int64_t y_min = _sel_line->top() * BUCKET_ELT_COUNT;
		int64_t y_max = _sel_line->bottom() * BUCKET_ELT_COUNT;

		if (_selection_sliders == nullptr) {
			_selection_sliders = _sliders_group->add_zoomed_selection_sliders(y_min, y_max);
			_selection_sliders->hide();
		}
		/* ::set_value implictly emits the signal slider_moved;
		 * what will lead to a selection update in PVFullParallelView
		 */
		_selection_sliders->set_value(y_min, y_max);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::cancel_and_wait_all_rendering
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::cancel_and_wait_all_rendering()
{
	if (_left_zone) {
		_left_zone->cancel_and_wait_all();
	}
	if (_right_zone) {
		_right_zone->cancel_and_wait_all();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::on_zoom_sliders_update
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::on_zoom_sliders_update(
    PVCombCol col,
    PVSlidersManager::id_t id,
    int64_t y_min,
    int64_t y_max,
    PVSlidersManager::ZoomSliderChange change)
{
	if (change == PVParallelView::PVSlidersManager::ZoomSliderNone) {
		return;
	}

	if ((col == _nraw_col) && (id == _sliders_group.get())) {
		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		double sld_min = y_min / (double)BUCKET_ELT_COUNT;
		double sld_max = y_max / (double)BUCKET_ELT_COUNT;
		double sld_dist = sld_max - sld_min;

		// computing the nearest range matching the discrete zoom rules
		double y_dist = std::round(
		    std::pow(2.0, (std::round(zoom_steps * std::log2(sld_dist)) / (double)zoom_steps)));

		if (y_dist != 0) {
			double screen_height = _zpview->get_viewport()->rect().height();
			auto wanted_alpha = PVCore::clamp<double>(y_dist / screen_height, 0., 1.);
			_wheel_value = (int)std::round(retrieve_wheel_value_from_alpha(wanted_alpha));
		} else {
			_wheel_value = max_wheel_value;
		}

		if (change == PVParallelView::PVSlidersManager::ZoomSliderMin) {
			sld_max = std::round(PVCore::clamp<double>(sld_min + y_dist, 0., image_height));
		} else if (change == PVParallelView::PVSlidersManager::ZoomSliderMax) {
			sld_min = std::round(PVCore::clamp<double>(sld_max - y_dist, 0., image_height));
		} else {
			throw std::runtime_error(
			    "did you expect to move the 2 zoom sliders at the same time?\n");
		}

		_zpview->center_on(0., 0.5 * (sld_min + sld_max));

		update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::on_zoom_sliders_del
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::on_zoom_sliders_del(PVCombCol col,
                                                                PVSlidersManager::id_t id)
{
	if ((col == _nraw_col) && (id == _sliders_group.get())) {
		if (_pending_deletion == false) {
			_pending_deletion = true;
			if (_zpview != nullptr) {
				_zpview->parentWidget()->close();
			}
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::on_zoomed_sel_sliders_del
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::on_zoomed_sel_sliders_del(PVCombCol col,
                                                                      PVSlidersManager::id_t /*id*/)
{
	if (col == _nraw_col) {
		_selection_sliders = nullptr;
		if (_zpview != nullptr) {
			_zpview->get_viewport()->update();
		}
	}
}
