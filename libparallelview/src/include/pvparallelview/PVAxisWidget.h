#ifndef PVPARALLELVIEW_PVAXISWIDGET_H
#define PVPARALLELVIEW_PVAXISWIDGET_H

#include <pvparallelview/PVAxisSlider.h>

#include <QGraphicsItem>

#include <vector>
#include <utility>

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

private:
	Picviz::PVAxis*                 _axis;
	QRectF                          _bbox;
	std::vector<PVAxisRangeSliders> _sliders;
};

}

#endif // PVPARALLELVIEW_PVAXISWIDGET_H
