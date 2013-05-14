
#ifndef PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H
#define PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H

#include <pvparallelview/PVZoomConverter.h>

namespace PVParallelView
{

/**
 * @class PVZoomConverterPowerOfTwo
 *
 * This class represents a PVZoomConverter where scales
 * are defined as a power of two of zooms.
 */

class PVZoomConverterPowerOfTwo : public PVZoomConverter
{
public:
	int scale_to_zoom(const qreal value) const override
	{
		return log2(value);
	}

	qreal zoom_to_scale(const int value) const override
	{
		return pow(2.0, value);
	}
};

}

#endif // PVPARALLELVIEW_PVZOOMCONVERTERPOWEROFTWO_H
