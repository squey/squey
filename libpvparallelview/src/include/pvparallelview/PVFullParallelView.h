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

#include <picviz/PVView.h>

namespace PVParallelView {

class PVFullParallelScene;
class PVRenderingJob;
class PVFullParallelScene;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
	friend class PVFullParallelScene;

public:
	PVFullParallelView(QWidget* parent = NULL);
	~PVFullParallelView();

public:
	void paintEvent(QPaintEvent *event);
	void resizeEvent(QResizeEvent *event);

	void set_total_line_number(uint32_t total_lines) { _total_lines = total_lines; }
	void set_selected_line_number(uint32_t selected_lines) { _selected_lines = selected_lines; }

signals:
	void new_zoomed_parallel_view(Picviz::PVView* view, int axis_index);

private:
	uint32_t _total_lines = 0;
	uint32_t _selected_lines = 0;
	bool     _first_resize;
};

}

#endif // __PVFULLPARALLELVIEW_H__
