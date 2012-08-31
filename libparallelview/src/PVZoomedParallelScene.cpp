/**
 * \file PVZoomedParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoomedParallelScene.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <QScrollBar>
#include <QtCore>

/**
 * TODO: keep ratio between the width of the 2 zones in full parallel view
 *       => stores the scale for each beta (for the left one and the right one)
 *
 * TODO: add a mechanism to store BCI codes for a given area in scene to avoid
 * extracting them from the quadree each time drawBackground is called.
 *
 * TODO: remove the limitation of 512 for the backend_image's width
 *
 * TODO: configure scene's view from the PVAxis
 *
 * TODO: finalize selection stuff
 *
 * TODO: calls to zone_drawing must be moved into tbb:task
 *
 * TODO: use _last_selection_square_screen_pos like in FullParallelScene
 *
 * TODO: check for a greater value for max_wheel_value
 *
 * TODO: do we limit the view size or not? If not, it remove the limitation on
 *       images width
 */

#define ZOOM_MODIFIER     Qt::NoModifier
#define PAN_MODIFIER      Qt::ControlModifier
#define SLOW_PAN_MODIFIER Qt::ShiftModifier

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene(PVParallelView::PVZoomedParallelView *zpview,
                                                             Picviz::FakePVView_p pvview_p,
                                                             zones_drawing_t &zones_drawing,
                                                             PVCol axis) :
	QGraphicsScene(zpview),
	_zpview(zpview),
	_pvview_p(pvview_p),
	_zones_drawing(zones_drawing),
	_selection(pvview_p->get_view_selection()),
	_axis(axis),
	_old_sb_pos(-1),
	_skip_update_zoom(true),
	_rendering_zone_number(0),
	_rendered_zone_count(0)
{
	_zpview->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	_zpview->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	_zpview->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	_zpview->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	_zpview->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	//_zpview->setMaximumWidth(1024);
	//_zpview->setMaximumHeight(1024);

	_selection_rect = new PVParallelView::PVSelectionSquareGraphicsItem(this);
	connect(_selection_rect, SIGNAL(commit_volatile_selection()),
	        this, SLOT(commit_volatile_selection_Slot()));

	_wheel_value = 0;

	setSceneRect(-512, 0, 1024, 1024);

	connect(_zpview->verticalScrollBar(), SIGNAL(valueChanged(int)),
	        this, SLOT(scrollbar_changed_Slot(int)));

	_rendering_job = new PVRenderingJob(this);
	connect(_rendering_job, SIGNAL(zone_rendered(int)),
	        this, SLOT(zone_rendered_Slot(int)));

	_left_zone.created = false;
	if (axis > 0) {
		_left_zone.bg_image = zones_drawing.create_image(image_width);
		_left_zone.sel_image = zones_drawing.create_image(image_width);
		++_rendering_zone_number;
		_left_zone.created = true;
	}

	_right_zone.created = false;
	if (axis < zones_drawing.get_zones_manager().get_number_zones()) {
		_right_zone.bg_image = zones_drawing.create_image(image_width);
		_right_zone.sel_image = zones_drawing.create_image(image_width);
		++_rendering_zone_number;
		_right_zone.created = true;
	}

	PVParallelView::PVZonesManager &zm = zones_drawing.get_zones_manager();
	connect(&zm, SIGNAL(filter_by_sel_finished(int, bool)),
	        this, SLOT(filter_by_sel_finished_Slot(int, bool)));

	_scroll_timer.setInterval(50);
	_scroll_timer.setSingleShot(true);
	connect(&_scroll_timer, SIGNAL(timeout()),
	        this, SLOT(scrollbar_timeout_Slot()));

	update_zoom();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::~PVZoomedParallelScene()
{
	delete _selection_rect;

	_rendering_job->cancel();
	_rendering_future.waitForFinished();
	_rendering_job->deleteLater();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
		_selection_rect_pos = event->scenePos();
	} else if (event->button() == Qt::RightButton) {
		_pan_reference_y = event->screenPos().y();
	}

	event->accept();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
		if (_selection_rect_pos == event->scenePos()) {
			// Remove selection
			_selection_rect->clear_rect();
		}
		commit_volatile_selection_Slot();
	}

	event->accept();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	if (event->buttons() == Qt::LeftButton) {
		// trace square area
		QPointF top_left(qMin(_selection_rect_pos.x(), event->scenePos().x()),
		                 qMin(_selection_rect_pos.y(), event->scenePos().y()));
		QPointF bottom_right(qMax(_selection_rect_pos.x(), event->scenePos().x()),
		                     qMax(_selection_rect_pos.y(), event->scenePos().y()));

		_selection_rect->update_rect(QRectF(top_left, bottom_right));
	} else if (event->buttons() == Qt::RightButton) {
		QScrollBar *sb = _zpview->verticalScrollBar();
		int delta = _pan_reference_y - event->screenPos().y();
		_pan_reference_y = event->screenPos().y();
		int v = sb->value();
		sb->setValue(v + delta);
	}

	event->accept();
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
				update_zoom(true);
			}
		} else {
			if (_wheel_value > 0) {
				--_wheel_value;
				update_zoom(false);
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
 * PVParallelView::PVZoomedParallelScene::invalidate_selection
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::invalidate_selection()
{
	/* The selection has changed ; all rendered zones must be considered
	 * as not rendered and the current running rendering job must be
	 * stopped. It is useless to wait for the job's end because
	 * ::update_display() will wait for its end when all needed calls to
	 * ::filter_by_sel_finished_Slot() will have been done.
	 */
	_rendered_zone_count = 0;
	_rendering_job->cancel();
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
	// transformation matrix is unneeded to draw the image
	QTransform t = painter->transform();
	painter->resetTransform();

	painter->fillRect(screen_rect, Qt::black);
	// draw the zones
	if (_left_zone.created) {
		painter->fillRect(_left_zone.area, Qt::black);
		painter->drawImage(_left_zone.pos, _left_zone.back_image);
	}

	if (_right_zone.created) {
		painter->fillRect(_right_zone.area, Qt::black);
		painter->drawImage(_right_zone.pos, _right_zone.back_image);
	}

	// the pen has to be saved too
	QPen old_pen = painter->pen();

	// draw axis
	QPen new_pen = QPen(Qt::white);
	new_pen.setWidth(PARALLELVIEW_AXIS_WIDTH);
	painter->setPen(new_pen);
	painter->drawLine(screen_center, 0, screen_center, screen_rect.height());

	// get back the painter's original state
	painter->setTransform(t);
	painter->setPen(old_pen);
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_display()
{
	if (_skip_update_zoom) {
		_skip_update_zoom = false;
		return;
	}

	double alpha = bbits_alpha_scale * pow(root_step, get_zoom_step());
	double beta = 1. / get_scale_factor();

	QRect screen_rect = _zpview->viewport()->rect();
	int screen_center = screen_rect.center().x();

	QRectF screen_rect_s = _zpview->mapToScene(screen_rect).boundingRect();
	QRectF view_rect = sceneRect().intersected(screen_rect_s);

	double pixel_height = (1UL << (32 - NBITS_INDEX)) / get_scale_factor();

	// the screen's upper limit in plotted coordinates system
	uint64_t y_min = view_rect.top() * (UINT32_MAX >> NBITS_INDEX);
	// the backend_image's lower limit in plotted coordinates system
	uint64_t y_lim = PVCore::clamp<uint64_t>(y_min + (1 << bbits) * alpha * pixel_height,
	                                         0ULL, 1ULL << 32);
	// the screen's lower limit in plotted coordinates system
	// y_max can not be greater than y_lim
	uint64_t y_max = PVCore::clamp<uint64_t>(y_min + screen_rect.height() * pixel_height,
	                                         0ULL, y_lim);

	int gap_y = (screen_rect_s.top() < 0)?round(-screen_rect_s.top()):0;

	if (_rendering_future.isRunning()) {
		_rendering_job->cancel();
		_rendering_future.waitForFinished();
	}
	_rendering_job->reset();
	_rendering_future = QtConcurrent::run<>([&, y_min, y_max, y_lim, alpha, beta,
	                                         gap_y, screen_rect, screen_center]
		{
			using namespace PVParallelView;

			if (_rendering_job->should_cancel()) {
				return;
			}

			BENCH_START(full_render);

			if (_left_zone.created && (_render_type == RENDER_ALL)) {
				BENCH_START(render);
				_zones_drawing.draw_zoomed_zone(*(_left_zone.bg_image), y_min, y_max, y_lim,
				                                _zoom_level, _axis - 1,
				                                &PVZoomedZoneTree::browse_bci_by_y2,
				                                alpha, beta, true);
				BENCH_END(render, "render left tile", 1, 1, 1, 1);

				if (_rendering_job->should_cancel()) {
					return;
				}

				BENCH_START(sel_render);
				_zones_drawing.draw_zoomed_zone_sel(*(_left_zone.sel_image),
				                                    y_min, y_max, y_lim, _selection,
				                                    _zoom_level, _axis - 1,
				                                    &PVZoomedZoneTree::browse_bci_sel_by_y2,
				                                    alpha, beta, true);
				BENCH_END(sel_render, "render selection of left tile", 1, 1, 1, 1);

				int gap_x = PARALLELVIEW_AXIS_WIDTH / 2;

				_left_zone.area = QRect(0, 0,
				                        screen_center + gap_x /* == width / 2 */,
				                        screen_rect.height());
				_left_zone.pos = QPoint(screen_center - gap_x - image_width, gap_y);

				if (_rendering_job->should_cancel()) {
					return;
				}
			}

			if (_right_zone.created && (_render_type == RENDER_ALL)) {
				BENCH_START(render);
				_zones_drawing.draw_zoomed_zone(*(_right_zone.bg_image), y_min, y_max, y_lim,
				                                _zoom_level, _axis,
				                                &PVZoomedZoneTree::browse_bci_by_y1,
				                                alpha, beta, false);
				BENCH_END(render, "render right tile", 1, 1, 1, 1);

				if (_rendering_job->should_cancel()) {
					return;
				}

				BENCH_START(sel_render);
				_zones_drawing.draw_zoomed_zone_sel(*(_right_zone.sel_image),
				                                    y_min, y_max, y_lim, _selection,
				                                    _zoom_level, _axis,
				                                    &PVZoomedZoneTree::browse_bci_sel_by_y1,
				                                    alpha, beta, false);
				BENCH_END(sel_render, "render selection of right tile", 1, 1, 1, 1);

				int value = 1 + screen_center + PARALLELVIEW_AXIS_WIDTH / 2;

				_right_zone.area = QRect(value, 0,
				                         screen_center /* == width / 2 */,
				                         screen_rect.height());
				_right_zone.pos = QPoint(value, gap_y);

				if (_rendering_job->should_cancel()) {
					return;
				}
			}

			BENCH_END(full_render, "full render of view", 1, 1, 1, 1);

			// the zone id is unused
			_rendering_job->zone_finished(0);
		});
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::resize_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::resize_display(const QSize &s)
{
	QImage image = QImage(s, QImage::Format_ARGB32);
	image.fill(Qt::black);

	if (_left_zone.created) {
		QImage &back_image = _left_zone.back_image;
		QImage image = QImage(image_width, s.height(), QImage::Format_ARGB32);
		image.fill(Qt::black);

		if (_left_zone.back_image.isNull() == false) {
			QPainter painter(&image);

			int old_half_width = back_image.width();
			int new_half_width = s.width() / 2;

			painter.drawImage(QPoint(new_half_width - old_half_width, 0),
			                  back_image);
		}

		back_image = image;
	}

	if (_right_zone.created) {
		QImage &back_image = _right_zone.back_image;
		QImage image = QImage(image_width, s.height(), QImage::Format_ARGB32);
		image.fill(Qt::black);

		if (_right_zone.back_image.isNull() == false) {
			QPainter painter(&image);

			int old_half_width = back_image.width();
			int new_half_width = s.width() / 2;

			painter.drawImage(QPoint(new_half_width - old_half_width, 0),
			                  back_image);
		}

		back_image = image;
	}

	update_all();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_zoom(bool in)
{
	if (_skip_update_zoom == false) {
		double image_scale = 1. - 1. / zoom_steps;
		if (in) {
			image_scale = 1. / image_scale;
		}

		qreal dy = _zpview->viewport()->rect().center().y();

		if (_left_zone.created) {
			QImage &back_image = _left_zone.back_image;

			QImage image = QImage(back_image.width(), back_image.height(),
			                      QImage::Format_ARGB32);
			image.fill(Qt::black);

			QPainter painter(&image);
			painter.setRenderHint(QPainter::SmoothPixmapTransform);

			QPoint v(image_width, dy);

			painter.translate(v);
			painter.scale(image_scale, image_scale);
			painter.translate(-v);
			painter.drawImage(QPoint(0, 0), back_image);

			back_image = image;
		}

		if (_right_zone.created) {
			QImage &back_image = _right_zone.back_image;

			QImage image = QImage(back_image.width(), back_image.height(),
			                      QImage::Format_ARGB32);
			image.fill(Qt::black);

			QPainter painter(&image);
			painter.setRenderHint(QPainter::SmoothPixmapTransform);

			QPoint v(0, dy);

			painter.translate(v);
			painter.scale(image_scale, image_scale);
			painter.translate(-v);
			painter.drawImage(QPoint(0, 0), back_image);

			back_image = image;
		}
	}

	/* make the next ::scrollbar_changed_Slot() call have no
	 * effect on all back_images. Changing the view's transformation
	 * matrix updates the scrollbar which will emit a valueChanged
	 * event; it will also cause a unwanted cout-out effect while
	 * zooming in.
	 */
	_old_sb_pos = -1;

	_zoom_level = get_zoom_level();
	double s = get_scale_factor();

	QMatrix mat;
	mat.scale(s, s);
	_zpview->setMatrix(mat);

	/* make sure the scene is always horizontally centered. And because
	 * of the selection rectangle, the scene's bounding box can change;
	 * so that its center could not be 0...
	 */
	QScrollBar *sb = _zpview->horizontalScrollBar();
	int64_t mid = ((int64_t)sb->maximum() + sb->minimum()) / 2;
	sb->setValue(mid);

	update_all();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot(int value)
{
	if (_old_sb_pos >= 0) {
		if (_left_zone.created) {
			QImage &back_image = _left_zone.back_image;

			QImage image = QImage(back_image.width(), back_image.height(),
			                      QImage::Format_ARGB32);
			image.fill(Qt::black);

			QPainter painter(&image);
			painter.drawImage(QPoint(0, _old_sb_pos - value),
			                  back_image);

			back_image = image;
		}

		if (_right_zone.created) {
			QImage &back_image = _right_zone.back_image;

			QImage image = QImage(back_image.width(), back_image.height(),
			                      QImage::Format_ARGB32);
			image.fill(Qt::black);

			QPainter painter(&image);
			painter.drawImage(QPoint(0, _old_sb_pos - value),
			                  back_image);

			back_image = image;
		}

		/* the full update is deferred only when the scrollbar is
		 * actived by the user; otherwise the update is done immediatly
		 * to avoid the impression that the refresh is slow.
		 */
		if (_zpview->verticalScrollBar()->isSliderDown()) {
			_scroll_timer.stop();
			_scroll_timer.start();
		} else {
			update_all();
		}
	}

	_old_sb_pos = value;
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::scrollbar_timeout_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::scrollbar_timeout_Slot()
{
	update_all();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::rendering_done_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::zone_rendered_Slot(int /*z*/)
{
	if (_left_zone.created) {
		QImage &image = _left_zone.back_image;
		image.fill(Qt::black);
		QPainter painter(&image);
		painter.setOpacity(0.25);
		painter.drawImage(0, 0, _left_zone.bg_image->qimage());
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _left_zone.sel_image->qimage());
	}

	if (_right_zone.created) {
		QImage &image = _right_zone.back_image;
		image.fill(Qt::black);
		QPainter painter(&image);
		painter.setOpacity(0.25);
		painter.drawImage(0, 0, _right_zone.bg_image->qimage());
		painter.setOpacity(1.0);
		painter.drawImage(0, 0, _right_zone.sel_image->qimage());
	}

	update();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::filter_by_sel_finished_Slot
 *****************************************************************************/

void
PVParallelView::PVZoomedParallelScene::filter_by_sel_finished_Slot(int zid,
                                                                   bool changed)
{
	if ((zid == _axis) && changed && _right_zone.created) {
		++_rendered_zone_count;
		if (_rendered_zone_count == _rendering_zone_number) {
			/* if all zone to render have been rendered,
			 * ::update_sel() can be called.
			 */
			update_sel();
		}
	} else if ((zid == (_axis - 1)) && changed && _left_zone.created) {
		++_rendered_zone_count;
		if (_rendered_zone_count == _rendering_zone_number) {
			/* if all zone to render have been rendered,
			 * ::update_sel() can be called.
			 */
			update_sel();
		}
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::commit_volatile_selection_Slot()
{
	_selection_rect->finished();
}
