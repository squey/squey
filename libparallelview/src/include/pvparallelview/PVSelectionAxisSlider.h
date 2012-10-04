
#ifndef PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H
#define PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H

#include <pvparallelview/PVAbstractAxisSlider.h>

namespace PVParallelView
{

class PVSelectionAxisSlider : public PVAbstractAxisSlider
{
public:
	PVSelectionAxisSlider(int64_t omin, int64_t omax, int64_t o,
	                      PVAxisSliderOrientation orientation = Min) :
		PVAbstractAxisSlider(omin, omax, o, orientation)
	{}

	virtual QRectF boundingRect () const;

	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
	                   QWidget *widget = 0);
};

}

#endif // PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H
