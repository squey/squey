/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H
#define PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H

#include <pvparallelview/PVAbstractAxisSlider.h>

namespace PVParallelView
{

class PVZoomedSelectionAxisSlider : public PVAbstractAxisSlider
{
  public:
	PVZoomedSelectionAxisSlider(int64_t omin,
	                            int64_t omax,
	                            int64_t o,
	                            PVAxisSliderOrientation orientation = Min)
	    : PVAbstractAxisSlider(omin, omax, o, orientation)
	{
	}

	QRectF boundingRect() const override;

	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMEDSELECTIONAXISSLIDER_H
