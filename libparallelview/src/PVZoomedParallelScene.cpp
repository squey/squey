/**
 * \file PVZoomedParallelScene.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZoomedParallelScene.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <QScrollBar>

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
 * TODO: add selection stuff
 *
 * TODO: parallelize zoom rendering
 *
 * TODO: make postponed and cancelable zoom rendering
 *
 * TODO: 
 */

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene
 *****************************************************************************/

PVParallelView::PVZoomedParallelScene::PVZoomedParallelScene(QWidget *parent,
                                                             zones_drawing_t &zones_drawing,
                                                             PVCol axis) :
	QGraphicsScene(parent),
	_zones_drawing(zones_drawing), _axis(axis),
	_old_sb_pos(-1)
{
	setBackgroundBrush(Qt::black);

	view()->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	view()->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	view()->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	view()->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view()->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

	//view()->setMaximumWidth(1024);
	//view()->setMaximumHeight(1024);

	_wheel_value = 0;

	setSceneRect(-512, 0, 1024, 1024);

	update_zoom();

	connect(view()->verticalScrollBar(), SIGNAL(valueChanged(int)),
	        this, SLOT(scrollbar_changed_Slot(int)));

	if (axis > 0) {
		_left_image = zones_drawing.create_image(image_width);
	}

	if (axis < zones_drawing.get_zones_manager().get_number_zones()) {
		_right_image = zones_drawing.create_image(image_width);
	}
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mousePressEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mousePressEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseReleaseEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseReleaseEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::mouseMoveEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::mouseMoveEvent(QGraphicsSceneMouseEvent */*event*/)
{
	// kill default behaviour of QGraphicsScene
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::wheelEvent
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::wheelEvent(QGraphicsSceneWheelEvent* event)
{
	if (event->modifiers() == Qt::ControlModifier) {
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
	} else if (event->modifiers() == Qt::ShiftModifier) {
		// precise panning
		QScrollBar *sb = view()->verticalScrollBar();
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
	} else if (event->modifiers() == Qt::NoModifier) {
		// default panning
		QScrollBar *sb = view()->verticalScrollBar();
		if (event->delta() > 0) {
			sb->triggerAction(QAbstractSlider::SliderSingleStepSub);
		} else {
			sb->triggerAction(QAbstractSlider::SliderSingleStepAdd);
		}
	}

	event->accept();
}


/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::drawBackground
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::drawBackground(QPainter *painter,
                                                           const QRectF &/*rect*/)
{
	QRect screen_rect = view()->viewport()->rect();
	int screen_center = screen_rect.center().x();

	// save the painter's state to restore it later because the scene
	// transformation matrix is unneeded to draw the image
	QTransform t = painter->transform();
	painter->resetTransform();

	// draw the full image
	if (_back_image.isNull() == false) {
		painter->drawImage(QPoint(0,0), _back_image);
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
	if (_back_image.isNull()) {
		// unneeded to render when the view has no size
		return;
	}

	QPainter painter(&_back_image);

	double alpha = bbits_alpha_scale * pow(root_step, get_zoom_step());
	double beta = 1. / get_scale_factor();

	QRect screen_rect = view()->viewport()->rect();
	int screen_center = screen_rect.center().x();

	QRectF screen_rect_s = view()->mapToScene(screen_rect).boundingRect();
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

	// TODO: the following code must be done in background
	if (_left_image.get() != nullptr) {
		BENCH_START(render);
		_zones_drawing.draw_zoomed_zone(*_left_image, y_min, y_max, y_lim,
		                                _zoom_level, _axis - 1,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y2,
		                                alpha, beta, true);
		BENCH_END(render, "render left tile", 1, 1, 1, 1);
		int gap_x = PARALLELVIEW_AXIS_WIDTH / 2;

		painter.fillRect(0, 0,
		                 screen_center + gap_x /* == width / 2 */, screen_rect.height(),
		                 Qt::black);
		painter.drawImage(QPoint(screen_center - gap_x - image_width, gap_y),
		                  _left_image->qimage());
	}

	if (_right_image.get() != nullptr) {
		BENCH_START(render);
		_zones_drawing.draw_zoomed_zone(*_right_image, y_min, y_max, y_lim,
		                                _zoom_level, _axis,
		                                &PVParallelView::PVZoomedZoneTree::browse_tree_bci_by_y1,
		                                alpha, beta, false);
		BENCH_END(render, "render right tile", 1, 1, 1, 1);

		int value = 1 + screen_center + PARALLELVIEW_AXIS_WIDTH / 2;

		painter.fillRect(value, 0,
		                 screen_center /* == width / 2 */, screen_rect.height(),
		                 Qt::black);
		painter.drawImage(QPoint(value, gap_y),
		                  _right_image->qimage());
	}

	update();
}


/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::resize_display
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::resize_display(const QSize &s)
{
	QImage image = QImage(s, QImage::Format_ARGB32);
	image.fill(Qt::black);

	if (_back_image.isNull() == false) {
		QPainter painter(&image);

		int old_half_width = _back_image.width() / 2;
		int new_half_width = s.width() / 2;

		// RH: is it better to center vertically the old image?
		painter.drawImage(QPoint(new_half_width - old_half_width, 0),
		                  _back_image);
	}

	_back_image = image;
	update_display();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::update_zoom
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::update_zoom(bool in)
{
	if (_back_image.isNull() == false) {
		QImage image = QImage(_back_image.width(), _back_image.height(),
		                      QImage::Format_ARGB32);
		image.fill(Qt::black);
		QPainter painter(&image);
		painter.setRenderHint(QPainter::SmoothPixmapTransform);

		double image_scale = 1. - 1. / zoom_steps;
		if (in) {
			image_scale = 1. / image_scale;
		}

		QPoint view_center = view()->viewport()->rect().center();

		painter.translate(view_center);
		painter.scale(image_scale, image_scale);
		painter.translate(-view_center);
		painter.drawImage(QPoint(0, 0), _back_image);

		_back_image = image;
	}

	/* make the next ::scrollbar_changed_Slot() call have no
	 * effect on _back_image. Changing the view's transformation
	 * matrix affect the scrollbar which will emit a valueChanged
	 * event; it will also cause a unwanted clipping effect while
	 * zooming in.
	 */
	_old_sb_pos = -1;

	_zoom_level = get_zoom_level();
	double s = get_scale_factor();

	QMatrix mat;
	mat.scale(s, s);

	view()->setMatrix(mat);

	update_display();
}

/*****************************************************************************
 * PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot
 *****************************************************************************/

void PVParallelView::PVZoomedParallelScene::scrollbar_changed_Slot(int value)
{
	if (_old_sb_pos >= 0) {
		QImage image = QImage(_back_image.width(), _back_image.height(),
		                      QImage::Format_ARGB32);
		image.fill(Qt::black);
		QPainter painter(&image);

		painter.drawImage(QPoint(0, _old_sb_pos - value),
		                  _back_image);

		_back_image = image;
		update_display();
	}

	_old_sb_pos = value;
}
