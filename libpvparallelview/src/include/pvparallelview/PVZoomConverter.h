
#ifndef PVPARALLELVIEW_PVZOOMCONVERTER_H
#define PVPARALLELVIEW_PVZOOMCONVERTER_H

namespace PVParallelView
{

/**
 * @class PVZoomConverter
 *
 * This class represents conversion between an abstract zoom value
 * and a scale factor.
 *
 * the function zoom_to_scale() must validate the following property:
 *
 * \f$z2s(x + y) = z2s(x) \times z2s(y)\f$
 *
 * and scale_to_zoom() is the invert of zoom_to_scale().
 */

class PVZoomConverter
{
public:
	/**
	 * Needed virtual DTOR to compile
	 */
	virtual ~PVZoomConverter()
	{}

public:
	/**
	 * retrieve a zoom value from a scale value
	 *
	 * @param value [in] the scale value
	 * @return the zoom value corresponding to @a value
	 */
	virtual int scale_to_zoom(const qreal value) const = 0;

	/**
	 * retrieve a scale value from a zoom value
	 *
	 * @param value [in] the zoom value
	 * @return the scale value corresponding to @a value
	 */
	virtual qreal zoom_to_scale(const int value) const = 0;
};

}

#endif // PVPARALLELVIEW_PVZOOMCONVERTER_H
