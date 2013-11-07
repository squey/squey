
#ifndef PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H
#define PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H

#include <pvparallelview/PVAbstractAxisSlider.h>

namespace PVParallelView
{

class PVZoomedSelectionAxisSlider : public PVAbstractAxisSlider
{
public:
	PVZoomedSelectionAxisSlider(int64_t omin, int64_t omax, int64_t o,
	                            PVAxisSliderOrientation orientation = Min) :
		PVAbstractAxisSlider(omin, omax, o, orientation)
	{}

	virtual QRectF boundingRect () const;

	virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
	                   QWidget *widget = 0);
};

}

#endif // PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H
