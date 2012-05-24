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

class PVAxisWidget : public QGraphicsItem
{
public:
	PVAxisWidget(Picviz::PVAxis *axis);

	~PVAxisWidget();

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
