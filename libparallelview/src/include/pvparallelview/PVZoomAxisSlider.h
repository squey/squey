
#ifndef PVPARALLELVIEW_PVZOOMAXISSLIDER
#define PVPARALLELVIEW_PVZOOMAXISSLIDER_H

#include <pvparallelview/PVAbstractAxisSlider.h>

namespace PVParallelView
{

class PVZoomAxisSlider : public PVAbstractAxisSlider
{
public:
	PVZoomAxisSlider(int64_t omin, int64_t omax, int64_t o,
	                 PVAxisSliderOrientation orientation = Min) :
		PVAbstractAxisSlider(omin, omax, o, orientation)
	{}

	virtual QRectF boundingRect () const;

	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
	                   QWidget *widget = 0);
};

}

#endif // PVPARALLELVIEW_PVZOOMAXISSLIDER_H
