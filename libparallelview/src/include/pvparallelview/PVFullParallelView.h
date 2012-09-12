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
class PVFullParallelScene;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT

public:
	PVFullParallelView(QWidget* parent = NULL);

	void paintEvent(QPaintEvent *event);

	void set_total_line_number(uint32_t total_lines) { _total_lines = total_lines; }
	void set_selected_line_number(uint32_t selected_lines) { _selected_lines = selected_lines; }

private:
	uint32_t _total_lines = 0;
	uint32_t _selected_lines = 0;
};

}

#endif // __PVFULLPARALLELVIEW_H__
