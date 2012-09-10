/**
 * \file PVFullParallelView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>

#include <QGLWidget>


PVParallelView::PVFullParallelView::PVFullParallelView(PVFullParallelScene* scene)
{
	setScene(scene);
	setViewport(new QWidget());
	resize(1920, 1600);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	show();
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
