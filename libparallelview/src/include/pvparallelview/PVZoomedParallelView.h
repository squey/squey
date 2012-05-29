#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

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
	                     PVCol axis);

	~PVZoomedParallelView();

	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

	void mousePressEvent(QGraphicsSceneMouseEvent *event);

	void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);

	void wheelEvent(QGraphicsSceneWheelEvent* event);


private:
	PVZonesDrawing   &_zones_drawing;
	PVZoomedZoneView *_left_zone;
	PVZoomedZoneView *_right_zone;
	PVCol             _axis;
	uint32_t          _y_position;
	qreal             _translation_start_y;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
