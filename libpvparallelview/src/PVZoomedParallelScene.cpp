/**
 * \file PVZoomedParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVAlgorithms.h>

#include <picviz/PVSelection.h>

#include <pvparallelview/PVAbstractAxisSlider.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoomedSelectionAxisSliders.h>
#include <pvparallelview/PVZoomedParallelScene.h>

#include <QMetaObject>
#include <QScrollBar>

/**
 * IDEA: keep ratio between the width of the 2 zones in full parallel view
 *       => stores the scale for each beta (for the left one and the right one)
 * RH: AG said this feature is not still planned (TODO -> IDEA)
 *
 * TODO: configure scene's view from the PVAxis
 *
 * TODO: finalize selection stuff
 *
 * TODO: make 0 be at the bottom of the view, not at the top
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

#define ZOOM_MODIFIER     Qt::NoModifier
#define PAN_MODIFIER      Qt::ControlModifier
#define SLOW_PAN_MODIFIER Qt::ShiftModifier

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
		Picviz::PVView_sp& pvview_sp,
		PVParallelView::PVSlidersManager_p sliders_manager_p,
		PVZonesProcessor& zp_sel,
		PVZonesProcessor& zp_bg,
		PVZonesManager const& zm,
		PVCol axis_index) :
	QGraphicsScene(zpview),
	_zpview(zpview),
	_pvview(*pvview_sp),
	_sliders_manager_p(sliders_manager_p),
	_zsu_obs(this),
	_zsd_obs(this),
	_axis_index(axis_index),
	_zm(zm),
	_pending_deletion(false),
	_left_zone(nullptr),
	_right_zone(nullptr),
	_zp_sel(zp_sel),
	_zp_bg(zp_bg),
	_selection_sliders(nullptr)
{
	_view_deleted = false;

	_zpview->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_zpview->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	_zpview->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	_zpview->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	_zpview->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	_zpview->setMaximumWidth(1024);
	_zpview->setMaximumHeight(1024);

	_axis_id = _pvview.get_axes_combination().get_axes_comb_id(axis_index);

	_selection_rect = new PVParallelView::PVSelectionSquareGraphicsItem(this);
	connect(_selection_rect, SIGNAL(commit_volatile_selection()),
			this, SLOT(commit_volatile_selection_Slot()));

	_wheel_value = 0;

	setSceneRect(-512, 0, 1024, 1024);

	connect(_zpview->verticalScrollBar(), SIGNAL(valueChanged(int)),
			this, SLOT(scrollbar_changed_Slot(int)));

	_sliders_group = new PVParallelView::PVSlidersGroup(sliders_manager_p, _axis_id);
	_sliders_group->setPos(0., 0.);
	_sliders_group->add_zoom_sliders(0, 1024);

	// the sliders must be over all other QGraphicsItems
	_sliders_group->setZValue(1.e42);

	addItem(_sliders_group);

	update_zones();

	PVHive::PVHive::get().register_func_observer(sliders_manager_p,
			_zsu_obs);

	PVHive::PVHive::get().register_func_observer(sliders_manager_p,
			_zsd_obs);

	_updateall_timer.setInterval(150);
	_updateall_timer.setSingleShot(true);
	connect(&_updateall_timer, SIGNAL(timeout()),
			this, SLOT(updateall_timeout_Slot()));

	PVHive::get().register_actor(pvview_sp, _view_actor);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene()
{
	if (_selection_rect) {
		delete _selection_rect;
		_selection_rect = nullptr;
	}

	if (!_view_deleted) {
		common::get_lib_view(_pvview)->remove_zoomed_view(this);
	}

	if (_sliders_group) {
		_sliders_group->delete_own_zoom_slider();
		delete _sliders_group;
		_sliders_group = nullptr;
	}

	if (_pending_deletion == false) {
		_pending_deletion = true;
		PVHive::call<FUNC(PVSlidersManager::del_zoom_sliders)>(_sliders_manager_p,
				_axis_id, _sliders_group);
	}

	if (_selection_sliders) {
		PVHive::call<FUNC(PVSlidersManager::del_zoomed_selection_sliders)>(_sliders_manager_p,
				_axis_id,
				_selection_sliders);
		_selection_sliders = nullptr;
	}

	if (_left_zone) {
		_left_zone->cancel_all();
		delete _left_zone;
	}

	if (_right_zone) {
		_right_zone->cancel_all();
		delete _right_zone;
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::RightButton) {
		_pan_reference_y = event->screenPos().y();
		event->accept();
	} else if (!_sliders_group->sliders_moving() && (event->button() == Qt::LeftButton)) {
		_selection_rect_pos = event->scenePos();
		event->accept();
	}

	// do item's hover stuff
	QGraphicsScene::mousePressEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	if (!_sliders_group->sliders_moving() && (event->button() == Qt::LeftButton)) {
		if (_selection_rect_pos == event->scenePos()) {
			// Remove selection
			_selection_rect->clear_rect();
			if (_selection_sliders) {
				PVHive::call<FUNC(PVSlidersManager::del_zoomed_selection_sliders)>(_sliders_manager_p,
						_axis_id,
						_selection_sliders);
				_selection_sliders = nullptr;
			}
		}
		commit_volatile_selection_Slot();
		_selection_rect->clear_rect();
		event->accept();
	}

	// do item's hover stuff
	QGraphicsScene::mouseReleaseEvent(event);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::RightButton) {
		QScrollBar *sb = _zpview->verticalScrollBar();
		int delta = _pan_reference_y - event->screenPos().y();
		_pan_reference_y = event->screenPos().y();
		int v = sb->value();
		sb->setValue(v + delta);
		event->accept();
	} else if (!_sliders_group->sliders_moving() && (event->buttons() == Qt::LeftButton)) {
		// trace square area
		QPointF top_left(_selection_rect_pos.x(),
				qMin(_selection_rect_pos.y(), event->scenePos().y()));
		QPointF bottom_right(_selection_rect_pos.x(),
				qMax(_selection_rect_pos.y(), event->scenePos().y()));

		_selection_rect->update_rect(QRectF(top_left, bottom_right));
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
		QScrollBar *sb = _zpview->verticalScrollBar();
		if (event->delta() > 0) {
			int v = sb->value();
			if (v > sb->minimum()) {
				sb->setValue(v - 1);
			}
		} else {
			int v = sb->value();
			if (v < sb->maximum()) {
				sb->setValue(v + 1);
			}
		}
	} else if (event->modifiers() == PAN_MODIFIER) {
		// panning
		QScrollBar *sb = _zpview->verticalScrollBar();
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

void PVParallelView::PVZoomedParallelScene::keyPressEvent(QKeyEvent *event)
{
	if (event->key() == Qt::Key_Space) {
		PVLOG_INFO("PVZoomedParallelScene: forcing full redraw\n");
		update_all();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zones
 *****************************************************************************/

bool PVParallelView::PVZoomedParallelScene::update_zones()
{
	PVCol axis = _pvview.get_axes_combination().get_index_by_id(_axis_id);

	if (axis == PVCOL_INVALID_VALUE) {
		if (_axis_index > get_zones_manager().get_number_of_managed_zones()) {
			/* a candidate can not be found to replace the old
			 * axis; the zoom view must be closed.
			 */
			return false;
		}

		/* the axis does not exist anymore, the one with the
		 * same index is used instead
		 */
		_axis_id = _pvview.get_axes_combination().get_axes_comb_id(_axis_index);

		// it's more simple to delete and recreate the sliders group
		_sliders_group->delete_own_zoom_slider();
		delete _sliders_group;

		_sliders_group = new PVParallelView::PVSlidersGroup(_sliders_manager_p, _axis_id);
		_sliders_group->setPos(0., 0.);
		_sliders_group->add_zoom_sliders(0, 1024);
	} else {
		/* the axes has only been moved, nothing special to do.
		*/
		_axis_index = axis;
	}

	// needed pixmap to create QGraphicsPixmapItem
	QPixmap dummy_pixmap;

	if (_left_zone) {
		removeItem(_left_zone->item);
		delete _left_zone->item;
		delete _left_zone;
		_left_zone = nullptr;
	}

	if (_right_zone) {
		removeItem(_right_zone->item);
		delete _right_zone->item;
		delete _right_zone;
		_right_zone = nullptr;
	}

	_renderable_zone_number = 0;

	if (_axis_index > 0) {
		_left_zone = new zone_desc_t;
		_left_zone->bg_image = common::backend().create_image(image_width, bbits);
		_left_zone->sel_image = common::backend().create_image(image_width, bbits);
		_left_zone->item = addPixmap(dummy_pixmap);

		++_renderable_zone_number;
	}

	if (_axis_index < get_zones_manager().get_number_of_managed_zones()) {
		_right_zone = new zone_desc_t;
		_right_zone->bg_image = common::backend().create_image(image_width, bbits);
		_right_zone->sel_image = common::backend().create_image(image_width, bbits);
		_right_zone->item = addPixmap(dummy_pixmap);

		++_renderable_zone_number;
	}

	if (_zpview->isVisible()) {
		update_all();
	}

	return true;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::drawBackground
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::drawBackground(QPainter *painter,
		const QRectF &/*rect*/)
{
	QRect screen_rect = _zpview->viewport()->rect();
	int screen_center = screen_rect.center().x();

	// save the painter's state to restore it later because the scene
	// transformation matrix is unneeded for what we have to draw
	QTransform t = painter->transform();
	painter->resetTransform();
	// the pen has to be saved too
	QPen old_pen = painter->pen();

	painter->fillRect(screen_rect, common::color_view_bg());

	// draw axis
	QPen new_pen = QPen(_pvview.get_axis(_axis_index).get_color().toQColor());

	new_pen.setWidth(PARALLELVIEW_AXIS_WIDTH);
	painter->setPen(new_pen);
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
 * PVParallelView::PVZoomedParallelScene::resize_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::resize_display()
{
	update_zoom();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_display()
{
	double alpha = bbits_alpha_scale * pow(root_step, get_zoom_step());
	double beta = 1. / get_scale_factor();

	QRect screen_rect = _zpview->viewport()->rect();
	int screen_center = screen_rect.center().x();

	QRectF screen_rect_s = _zpview->mapToScene(screen_rect).boundingRect();
	QRectF view_rect = sceneRect().intersected(screen_rect_s);

	double pixel_height = (1UL << (32 - NBITS_INDEX)) / get_scale_factor();

	// the screen's upper limit in plotted coordinates system
	uint64_t y_min = view_rect.top() * BUCKET_ELT_COUNT;
	// the backend_image's lower limit in plotted coordinates system
	uint64_t y_lim = PVCore::clamp<uint64_t>(y_min + (1 << bbits) * alpha * pixel_height,
			0ULL, 1ULL << 32);
	// the screen's lower limit in plotted coordinates system
	// y_max can not be greater than y_lim
	uint64_t y_max = PVCore::clamp<uint64_t>(y_min + screen_rect.height() * pixel_height,
			0ULL, y_lim);

	PVHive::call<FUNC(PVSlidersManager::update_zoom_sliders)>(_sliders_manager_p,
			_axis_id, _sliders_group,
			y_min, y_max,
			PVParallelView::PVSlidersManager::ZoomSliderNone);
	_last_y_min = y_min;
	_last_y_max = y_max;

	_next_beta = beta;

	int gap_y = (screen_rect_s.top() < 0)?round(-screen_rect_s.top()):0;

	_renderable_zone_number = 0;
	if (_left_zone) {
		if (_render_type == RENDER_ALL) {
			_left_zone->cancel_last_bg();
		}
		else {
			// If the background is rendering, we need to wait for one more rendering!
			_renderable_zone_number += (bool) _left_zone->last_zr_bg;
		}
		_left_zone->cancel_last_sel();

		QPoint npos(screen_center - image_width - axis_half_width, gap_y);

		_left_zone->next_pos = _zpview->mapToScene(npos);
	}

	if (_right_zone) {
		if (_render_type == RENDER_ALL) {
			_right_zone->cancel_last_bg();
		}
		else {
			// If the background is rendering, we need to wait for one more rendering!
			_renderable_zone_number += (bool) _right_zone->last_zr_bg;
		}
		_right_zone->cancel_last_sel();

		QPoint npos(screen_center + axis_half_width + 1, gap_y);

		_right_zone->next_pos = _zpview->mapToScene(npos);
	}


	int zoom_level = get_zoom_level();

#if 0
	// a second to slow the rendering
	for(int i = 0; i < 20; ++i) {
		usleep(50000);
	}
#endif

	if (_left_zone) {
		if (_render_type == RENDER_ALL) {
			_renderable_zone_number++;
			PVZoneRendering_p<bbits>
				zr(new PVZoneRendering<bbits>(left_zone_id(),
				                              [&,y_min,y_max,y_lim,zoom_level,beta](PVZoneID const z,
				                                                                    PVCore::PVHSVColor const* colors,
				                                                                    PVBCICode<bbits>* codes)
				                              {
					                              zzt_context_t ctxt;
					                              return get_zztree(z).browse_bci_by_y2(ctxt,
					                                                                    y_min, y_max, y_lim,
					                                                                    zoom_level, image_width,
					                                                                    colors, codes, beta);
				                              },
				                              *_left_zone->bg_image,
				                              0, // x_start
				                              image_width,
				                              alpha, // zoom_y
				                              true)); // reversed

			connect_zr(zr.get(), "zr_finished");
			_left_zone->last_zr_bg = zr;
			_zp_bg.add_job(zr);
		}

		_renderable_zone_number++;
		PVZoneRendering_p<bbits>
			zr(new PVZoneRendering<bbits>(left_zone_id(),
			                              [&,y_min,y_max,y_lim,zoom_level,beta](PVZoneID const z,
			                                                                    PVCore::PVHSVColor const* colors,
			                                                                    PVBCICode<bbits>* codes)
			                              {
				                              zzt_context_t ctxt;
				                              return get_zztree(z).browse_bci_sel_by_y2(ctxt,
					                                                                y_min, y_max,y_lim,
					                                                                real_selection(),
					                                                                zoom_level, image_width,
					                                                                colors, codes, beta);
			                              },
			                              *_left_zone->sel_image,
			                              0, // x_start
			                              image_width,
			                              alpha, // zoom_y
			                              true)); // reversed

		connect_zr(zr.get(), "zr_finished");
		_left_zone->last_zr_sel = zr;
		_zp_sel.add_job(zr);
	}

	if (_right_zone) {
		if (_render_type == RENDER_ALL) {
			_renderable_zone_number++;
			PVZoneRendering_p<bbits>
				zr(new PVZoneRendering<bbits>(right_zone_id(),
				                              [&,y_min,y_max,y_lim,zoom_level,beta](PVZoneID const z,
				                                                                    PVCore::PVHSVColor const* colors,
				                                                                    PVBCICode<bbits>* codes)
				                              {
					                              zzt_context_t ctxt;
					                              return get_zztree(z).browse_bci_by_y1(ctxt,
					                                                                    y_min, y_max, y_lim,
					                                                                    zoom_level, image_width,
					                                                                    colors, codes, beta);
				                              },
				                              *_right_zone->bg_image,
				                              0, // x_start
				                              image_width,
				                              alpha, // zoom_y
				                              false)); // reversed

			connect_zr(zr.get(), "zr_finished");
			_right_zone->last_zr_bg = zr;
			_zp_bg.add_job(zr);
		}

		_renderable_zone_number++;
		PVZoneRendering_p<bbits>
			zr(new PVZoneRendering<bbits>(right_zone_id(),
			                              [&,y_min,y_max,y_lim,zoom_level,beta](PVZoneID const z,
			                                                                    PVCore::PVHSVColor const* colors,
			                                                                    PVBCICode<bbits>* codes)
			                              {
				                              zzt_context_t ctxt;
				                              return get_zztree(z).browse_bci_sel_by_y1(ctxt,
					                                                                y_min, y_max,y_lim,
					                                                                real_selection(),
					                                                                zoom_level, image_width,
					                                                                colors, codes, beta);
			                              },
			                              *_right_zone->sel_image,
			                              0, // x_start
			                              image_width,
			                              alpha, // zoom_y
			                              false)); // reversed

		connect_zr(zr.get(), "zr_finished");
		_right_zone->last_zr_sel = zr;
		_zp_sel.add_job(zr);
	}
}

void PVParallelView::PVZoomedParallelScene::connect_zr(PVZoneRendering<bbits>* zr, const char* slot)
{
	zr->set_render_finished_slot(this, slot);
}

void PVParallelView::PVZoomedParallelScene::zr_finished(PVZoneRenderingBase_p zr, int zone_id)
{
	assert(is_zone_rendered(zone_id));
	assert(QThread::currentThread() == this->thread());
	bool zr_catch = true;

	if (zone_id == left_zone_id()) {
		if (_left_zone->last_zr_sel == zr) {
			_left_zone->last_zr_sel.reset();
		}
		else if (_left_zone->last_zr_bg == zr) {
			_left_zone->last_zr_bg.reset();
		}
		else {
			zr_catch = false;
		}
	}
	else {
		if (_right_zone->last_zr_sel == zr) {
			_right_zone->last_zr_sel.reset();
		}
		else if (_right_zone->last_zr_bg == zr) {
			_right_zone->last_zr_bg.reset();
		}
		else {
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

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_zoom()
{
	double scale_factor = get_scale_factor();

	QMatrix mat;
	mat.scale(scale_factor, scale_factor);
	_zpview->setMatrix(mat);

	/* make sure the scene is always horizontally centered. And because
	 * of the selection rectangle, the scene's bounding box can change;
	 * so that its center could not be 0...
	 */
	QScrollBar *sb = _zpview->horizontalScrollBar();
	int64_t mid = ((int64_t)sb->maximum() + sb->minimum()) / 2;
	sb->setValue(mid);

	/* now, it's time to tell each QGraphicsPixmapItem where it must be.
	*/
	if (_left_zone) {
		QPointF p = _left_zone->item->pos();
		double screen_left_x = _zpview->viewport()->rect().center().x();

		/* the image's size depends on its last computed beta factor
		 * and the current scale factor (it's obvious to prove but I
		 * have no time to:)
		 */
		screen_left_x -= image_width * (_current_beta * scale_factor);
		screen_left_x -= axis_half_width;

		/* mapToScene use ints, so, to avoid artefacts (due to the cast
		 * from double to int (a floor)), a rounded value is needed.
		 */
		if (screen_left_x < 0) {
			screen_left_x -= 0.5;
		} else {
			screen_left_x += 0.5;
		}

		QPointF np = _zpview->mapToScene(QPoint(screen_left_x, 0));

		_left_zone->item->setPos(QPointF(np.x(), p.y()));
	}

	if (_right_zone) {
		QPointF p = _right_zone->item->pos();
		int screen_right_x = _zpview->viewport()->rect().center().x() + axis_half_width + 1;

		/* mapToScene use ints, so, to avoid artefacts (due to the cast
		 * from double to int (a floor)), a rounded value is needed.
		 *
		 * RH: I am not sure about doing like left zone but I notice
		 *     less artefacts
		 */
		if (screen_right_x < 0) {
			screen_right_x -= 0.5;
		} else {
			screen_right_x += 0.5;
		}

		QPointF np = _zpview->mapToScene(QPoint(screen_right_x, 0));

		_right_zone->item->setPos(QPointF(np.x(), p.y()));
	}

	_updateall_timer.start();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot(int /*value*/)
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
	QMetaObject::invokeMethod(this, "update_all", Qt::QueuedConnection);
}

void PVParallelView::PVZoomedParallelScene::update_new_selection_async()
{
	QMetaObject::invokeMethod(this, "update_sel", Qt::QueuedConnection);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::all_rendering_done
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::all_rendering_done()
{
	QImage image(image_width, image_height, QImage::Format_ARGB32);
	QPainter painter(&image);

	_current_beta = _next_beta;

	if (_left_zone) {
		image.fill(Qt::transparent);

		painter.setOpacity(0.25);
		painter.drawImage(0, 0, _left_zone->bg_image->qimage());
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _left_zone->sel_image->qimage());

		_left_zone->item->setPos(_left_zone->next_pos);
		_left_zone->item->setScale(_current_beta);
		_left_zone->item->setPixmap(QPixmap::fromImage(image));
	}

	if (_right_zone) {
		image.fill(Qt::transparent);

		painter.setOpacity(0.25);
		painter.drawImage(0, 0, _right_zone->bg_image->qimage());
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _right_zone->sel_image->qimage());

		_right_zone->item->setPos(_right_zone->next_pos);
		_right_zone->item->setScale(_current_beta);
		_right_zone->item->setPixmap(QPixmap::fromImage(image));
	}

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot()
{
	_selection_rect->finished();

	Picviz::PVSelection &vol_sel = _pvview.get_volatile_selection();
	vol_sel.select_none();

	if (_selection_rect->is_null()) {
		// force selection update
		_view_actor.call<FUNC(Picviz::PVView::set_square_area_mode)>(Picviz::PVStateMachine::AREA_MODE_SET_WITH_VOLATILE);
		_view_actor.call<FUNC(Picviz::PVView::commit_volatile_in_floating_selection)>();
		_view_actor.call<FUNC(Picviz::PVView::process_real_output_selection)>();
	} else {
		int64_t y_min = _selection_rect->top() * BUCKET_ELT_COUNT;
		int64_t y_max = _selection_rect->bottom() * BUCKET_ELT_COUNT;

		if (_selection_sliders == nullptr) {
			_selection_sliders = _sliders_group->add_zoomed_selection_sliders(y_min, y_max);
		}
		/* ::set_value implictly emits the signal slider_moved;
		 * what will lead to a selection update in PVFullParallelView
		 */
		_selection_sliders->set_value(y_min, y_max);
	}
}

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
 * PVParallelView::PVZoomedParallelScene::zoom_sliders_update_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::zoom_sliders_update_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);
	PVParallelView::PVSlidersManager::ZoomSliderChange change = std::get<4>(args);

	if (change == PVParallelView::PVSlidersManager::ZoomSliderNone) {
		return;
	}

	if ((axis_id == _parent->_axis_id) && (id == _parent->_sliders_group)) {
		int64_t y_min = std::get<2>(args);
		int64_t y_max = std::get<3>(args);

		if (y_max < y_min) {
			std::swap(y_min, y_max);
		}

		double sld_min = y_min / (double)BUCKET_ELT_COUNT;
		double sld_max = y_max / (double)BUCKET_ELT_COUNT;
		double sld_dist = sld_max - sld_min;

		// computing the nearest range matching the discrete zoom rules
		double y_dist = round(pow(2.0, (round(zoom_steps * log2(sld_dist)) / (double)zoom_steps)));

		if (y_dist != 0) {
			double screen_height = _parent->_zpview->viewport()->rect().height();
			double wanted_alpha = PVCore::clamp<double>(y_dist / screen_height, 0., 1.);
			_parent->_wheel_value = (int)round(_parent->retrieve_wheel_value_from_alpha(wanted_alpha));
		} else {
			_parent->_wheel_value = max_wheel_value;
		}

		if (change == PVParallelView::PVSlidersManager::ZoomSliderMin) {
			sld_max = round(PVCore::clamp<double>(sld_min + y_dist,
						0., image_height));
		} else if (change == PVParallelView::PVSlidersManager::ZoomSliderMax) {
			sld_min = round(PVCore::clamp<double>(sld_max - y_dist,
						0., image_height));
		} else {
			PVLOG_WARN("did you expect to move the 2 zoom sliders at the same time?\n");
			return;
		}

		_parent->_zpview->centerOn(0., 0.5 * (sld_min + sld_max));

		_parent->update_zoom();
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::zoom_sliders_del_obs::update
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::zoom_sliders_del_obs::update(arguments_deep_copy_type const& args) const
{
	const axis_id_t &axis_id = std::get<0>(args);
	PVSlidersManager::id_t id = std::get<1>(args);

	if ((axis_id == _parent->_axis_id) && (id == _parent->_sliders_group)) {
		if (_parent->_pending_deletion == false) {
			_parent->_pending_deletion = true;
			_parent->_zpview->parentWidget()->close();
		}
	}
}
