#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneWheelEvent>

#include <pvbase/types.h>

namespace PVParallelView {

class PVZonesDrawing;

class PVZoomedZoneView;

class PVZoomedParallelView : public QGraphicsScene
{
public:
	PVZoomedParallelView(QObject* parent, PVParallelView::PVZonesDrawing &zones_drawing,
	                     int top, int bottom,
	                     PVCol axis);

	~PVZoomedParallelView();

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void mousePressEvent(QGraphicsSceneMouseEvent *event);

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);

private:
	QGraphicsView* view()
	{
		return (QGraphicsView*) parent();
	}

private:
	PVZonesDrawing   &_zones_drawing;
	PVZoomedZoneView *_left_zone;
	PVZoomedZoneView *_right_zone;
	PVCol             _axis;
	int               _top;
	int               _bottom;
	qreal             _translation_start_y;
	QImage            _left_images[4];
	QImage            _right_images[4];
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
