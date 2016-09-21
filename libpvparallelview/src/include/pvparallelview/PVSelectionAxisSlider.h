/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H
#define PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H

#include <pvparallelview/PVAbstractAxisSlider.h>

namespace PVParallelView
{

class PVSelectionAxisSlider : public PVAbstractAxisSlider
{
  public:
	PVSelectionAxisSlider(int64_t omin,
	                      int64_t omax,
	                      int64_t o,
	                      PVAxisSliderOrientation orientation = Min)
	    : PVAbstractAxisSlider(omin, omax, o, orientation)
	{
	}

	QRectF boundingRect() const override;

	void
	paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0) override;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSELECTIONAXISSLIDER_H
