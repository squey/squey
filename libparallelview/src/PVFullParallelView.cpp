/**
 * \file PVFullParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>

#include <QGLWidget>


PVParallelView::PVFullParallelView::PVFullParallelView(QWidget* parent):
	QGraphicsView(parent)
{
	setViewport(new QWidget());
	resize(800, 600);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	//show();
}

PVParallelView::PVFullParallelView::~PVFullParallelView()
{
	PVLOG_INFO("In PVFullParallelView destructor\n");
	if (scene()) {
		scene()->deleteLater();
	}
}

void PVParallelView::PVFullParallelView::paintEvent(QPaintEvent *event)
{
    QGraphicsView::paintEvent(event);

    QPainter painter(viewport());
	painter.setPen(QColor(0x16, 0xe8, 0x2a));
	QString count = QString("%L1 / %L2").arg(_selected_lines).arg(_total_lines);
	painter.drawText(width() - QFontMetrics(painter.font()).width(count) - 20, 20, count);
	painter.end();
}

void PVParallelView::PVFullParallelView::resizeEvent(QResizeEvent *event)
{
	QGraphicsView::resizeEvent(event);

	PVParallelView::PVFullParallelScene *fps = (PVParallelView::PVFullParallelScene*)scene();
	if(fps != nullptr) {
		fps->update_viewport();
		fps->update_scene();
		fps->update_all();

		/* to force the view to be always at the top. Otherwise,
		 * resizing the window to a smallet size automatically translates
		 * the view in a wrong way.
		 */
		verticalScrollBar()->setValue(verticalScrollBar()->minimum());
	}
}
