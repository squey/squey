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

// forward declaration
class PVZoomedParallelScene;

/**
 * @class PVZoomedParallelView
 *
 * A derived class of QGraphicsView to use when displaying a zoom view of parallel coordinates
 * representation of events.
 */
class PVZoomedParallelView : public QGraphicsView
{
public:
	/**
	 * Constructor
	 *
	 * @param parent parent widget
	 */
	PVZoomedParallelView(QWidget *parent = nullptr);

	/**
	 * Destructor
	 */
	~PVZoomedParallelView();

	/**
	 * Overload method when a resize event occurs.
	 *
	 * @param event then resize event
	 */
	void resizeEvent(QResizeEvent *event);
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
