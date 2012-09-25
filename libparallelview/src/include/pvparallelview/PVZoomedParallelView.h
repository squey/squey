/**
 * \file PVZoomedParallelView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <QGraphicsView>

namespace PVParallelView
{

class PVZoomedParallelScene;

class PVZoomedParallelView : public QGraphicsView
{
public:
	PVZoomedParallelView(QWidget *parent = nullptr);

	~PVZoomedParallelView();

	void resizeEvent(QResizeEvent *event);
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
