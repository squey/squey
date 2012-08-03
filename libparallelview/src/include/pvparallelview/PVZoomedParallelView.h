/**
 * \file PVZoomedParallelView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <QGraphicsView>
#include <QResizeEvent>

//#include <pvparallelview/PVZoomedTiler.h>

namespace PVParallelView {

class PVZoomedParallelView : public QGraphicsView
{
	Q_OBJECT

private:

	void resizeEvent(QResizeEvent* event)
	{
#if 0
		// forcing update of the scene's sceneRect
		((PVZoomedTiler*)scene())->update_scene_space();

		// re-centering the view
		QRectF r = mapToScene(rect()).boundingRect();
		centerOn(QPointF(0., r.center().y()));

		// and propagating the event (it is required)
		QGraphicsView::resizeEvent(event);
#endif
	}

};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
