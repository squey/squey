/**
 * \file PVFullParallelView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __PVFULLPARALLELVIEW_H__
#define __PVFULLPARALLELVIEW_H__

#include <QGraphicsView>
#include <QScrollBar>
#include <QFuture>
#include <QFontMetrics>
#include <QLocale>

#include <iostream>

#include <pvparallelview/common.h>

namespace PVParallelView {

class PVFullParallelScene;
class PVRenderingJob;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
public:
	PVFullParallelView()
	{
		setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	}

	void paintEvent(QPaintEvent *event)
	{
	    QGraphicsView::paintEvent(event);

	    QPainter painter(viewport());
		painter.setPen(QColor(0x16, 0xe8, 0x2a));
		QString count = QString("%L1 / %L2").arg(_selected_lines).arg(_total_lines);
		painter.drawText(width() - QFontMetrics(painter.font()).width(count) - 20, 20, count);
		painter.end();
	}

	void set_total_line_number(uint32_t total_lines) { _total_lines = total_lines; }
	void set_selected_line_number(uint32_t selected_lines) { _selected_lines = selected_lines; }

private:
	uint32_t _total_lines = 0;
	uint32_t _selected_lines = 0;
};

}

#endif // __PVFULLPARALLELVIEW_H__
