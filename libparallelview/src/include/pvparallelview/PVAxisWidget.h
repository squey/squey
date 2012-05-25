#ifndef PVPARALLELVIEW_PVAXISWIDGET_H
#define PVPARALLELVIEW_PVAXISWIDGET_H

#include <vector>
#include <utility>

#include <QGraphicsItem>

#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisSlider.h>

// Used to draw the axis out of the image zone
#define PVAW_CST 8

namespace Picviz
{
class PVAxis;
}

namespace PVParallelView
{

typedef std::pair<PVAxisSlider*, PVAxisSlider*> PVAxisRangeSliders;

// TODO: Maybe rename this class to PVAxisGraphicsItem since it's more a QGraphicsItem than a QWidget...

class PVAxisWidget : public QGraphicsItemGroup
{
public:
	PVAxisWidget(Picviz::PVAxis *axis);

	QRectF boundingRect () const;

	void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

	void add_range_sliders(uint32_t y1, uint32_t y2);

	QPointF map_from_scene(QPointF point) const
	{
		// Use a QPointF because mapFromScene called with a QRectF returns a QPolygon...
		QPointF mapped_point = mapFromScene(point);
		return QPointF(mapped_point.x() - PVParallelView::AxisWidth, mapped_point.y() + PVAW_CST);
	}

	QPointF map_to_scene(QPointF point) const
	{
		// Use a QPointF because mapFromScene called with a QRectF returns a QPolygon...

		point.rx() += PVParallelView::AxisWidth;
		point.ry() -= PVAW_CST;
		QPointF mapped_point = mapToScene(point);
		return QPointF(mapped_point.x(), mapped_point.y());
	}

private:
	Picviz::PVAxis*                 _axis;
	QRectF                          _bbox;
	std::vector<PVAxisRangeSliders> _sliders;
};

}

#endif // PVPARALLELVIEW_PVAXISWIDGET_H
