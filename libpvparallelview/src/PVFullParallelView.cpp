/**
 * \file PVFullParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVParallelView.h>

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::PVFullParallelView
 *
 *****************************************************************************/
PVParallelView::PVFullParallelView::PVFullParallelView(QWidget* parent):
	QGraphicsView(parent),
	_first_resize(true)
{
	setCursor(Qt::CrossCursor);
	setViewport(new QWidget());
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	setMinimumHeight(300);
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::~PVFullParallelView
 *
 *****************************************************************************/
PVParallelView::PVFullParallelView::~PVFullParallelView()
{
	PVLOG_DEBUG("In PVFullParallelView destructor\n");
	if (scene()) {
		scene()->deleteLater();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::paintEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelView::paintEvent(QPaintEvent *event)
{
    QGraphicsView::paintEvent(event);

    QPainter painter(viewport());
	QPen pen(QColor(0x16, 0xe8, 0x2a));
	painter.setPen(pen);
	
	// We set the string that gives the number of selected events, % and total number
	QString count = QString("%L1 (%2 %) / %L3").arg(_selected_lines).arg((uint32_t) (100.0*(double)_selected_lines/(double)_total_lines)).arg(_total_lines);
	
	painter.drawText(width() - QFontMetrics(painter.font()).width(count) - 20, 20, count);

#ifdef PICVIZ_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter.setPen(pen);

		const QPolygonF scene_rect = mapFromScene(scene()->sceneRect());
		painter.setPen(QColor(0xFF, 0, 0));
		painter.setBrush(QColor(0xFF, 0, 0, 40));
		painter.drawPolygon(scene_rect);

		pen.setColor(QColor(0xf6, 0xf2, 0x40));
		painter.setPen(pen);
		const QPolygonF items_rect = mapFromScene(scene()->itemsBoundingRect());
		painter.drawPolygon(items_rect);
	}
#endif
	
	painter.end();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::resizeEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelView::resizeEvent(QResizeEvent *event)
{
	QGraphicsView::resizeEvent(event);

	PVParallelView::PVFullParallelScene *fps = (PVParallelView::PVFullParallelScene*)scene();
	if(fps != nullptr) {
		fps->update_viewport();
		if(_first_resize) {
			_first_resize = false;
			fps->reset_zones_layout_to_default();
		} else {
			fps->update_scene(true);
		}
		fps->update_all_with_timer();

		/* to force the view to be always at the top. Otherwise,
		 * resizing the window to a smaller size automatically translates
		 * the view in a wrong way.
		 */
		verticalScrollBar()->setValue(verticalScrollBar()->minimum());
	}
}
