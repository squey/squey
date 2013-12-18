/**
 * \file PVFullParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVParallelView.h>

#include <QPaintEvent>
#include <QApplication>

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::PVFullParallelView
 *
 *****************************************************************************/
PVParallelView::PVFullParallelView::PVFullParallelView(QWidget* parent):
	QGraphicsView(parent),
	_first_resize(true)
{
	viewport()->setCursor(Qt::CrossCursor);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	setMinimumHeight(300);

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("full parallel view's help",
	                               ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-sel-rect-simple");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-full-parallel-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-shortcuts-full-parallel-view");
	_help_widget->finalizeText();
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
}

void PVParallelView::PVFullParallelView::drawForeground(QPainter* painter, const QRectF& rect)
{
	// Get back in viewport's coordinates system
	painter->save();
	painter->resetTransform();

	QRectF rect_view = mapFromScene(rect).boundingRect();

	QPen pen(QColor(0x16, 0xe8, 0x2a));
	painter->setPen(pen);
	
	QString count = QString("%L1 (%2 %) / %L3").arg(_selected_events_number).arg((uint32_t) (100.0*(double)_selected_events_number/(double)_total_events_number)).arg(_total_events_number);

	// The "count" string is drawn only if necessary
	QFontMetrics fm(painter->font());
	QSize text_size = fm.size(Qt::TextSingleLine, count);
	QPoint text_pos(width() - text_size.width() - 20, 20);
	if (QRectF(text_pos, text_size).intersects(rect_view)) {
		painter->drawText(text_pos, count);
	}

#ifdef PICVIZ_DEVELOPER_MODE
	if (common::show_bboxes()) {
		painter->setPen(pen);

		const QPolygonF scene_rect = mapFromScene(scene()->sceneRect());
		painter->setPen(QColor(0xFF, 0, 0));
		painter->setBrush(QColor(0xFF, 0, 0, 40));
		painter->drawPolygon(scene_rect);

		pen.setColor(QColor(0xf6, 0xf2, 0x40));
		painter->setPen(pen);
		const QPolygonF items_rect = mapFromScene(scene()->itemsBoundingRect());
		painter->drawPolygon(items_rect);

		pen.setColor(QColor(0x00, 0x00, 0xFF));
		painter->setPen(pen);
		QList<QGraphicsItem*> pixmaps = scene()->items();
		for (QGraphicsItem* p: pixmaps) {
			if (dynamic_cast<QGraphicsPixmapItem*>(p)) {
				painter->drawPolygon(mapFromScene(p->sceneBoundingRect()));
			}
		}
	}
#endif

	painter->restore();
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
			fps->update_scene(false);
		}
		fps->scale_all_zones_images();
		fps->update_all_with_timer();

		/* to force the view to be always at the top. Otherwise,
		 * resizing the window to a smaller size automatically translates
		 * the view in a wrong way.
		 */
		verticalScrollBar()->setValue(verticalScrollBar()->minimum());
	}
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::enterEvent
 *****************************************************************************/

void PVParallelView::PVFullParallelView::enterEvent(QEvent*)
{
	setFocus(Qt::MouseFocusReason);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::leaveEvent
 *****************************************************************************/

void PVParallelView::PVFullParallelView::leaveEvent(QEvent*)
{
	clearFocus();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::fake_mouse_move
 *****************************************************************************/

void PVParallelView::PVFullParallelView::fake_mouse_move()
{
	QMouseEvent e((QEvent::MouseMove),
	              mapFromGlobal(QCursor::pos()),
	              Qt::NoButton,
	              Qt::NoButton,
	              Qt::NoModifier);
	QApplication::sendEvent(viewport(), &e);
}
