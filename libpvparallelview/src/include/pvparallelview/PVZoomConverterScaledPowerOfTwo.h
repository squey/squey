
#ifndef PVPARALLELVIEW_PVZOOMCONVERTERSCALEDPOWEROFTWO_H
#define PVPARALLELVIEW_PVZOOMCONVERTERSCALEDPOWEROFTWO_H

namespace PVParallelView
{

/**
 * @class PVZoomConverterScaledPowerOfTwo
 *
 * This class is a variant of PVZoomConverterPowerOfTwo.
 *
 * As a scale of 2.0 can be too fast when it correspond to one mouse's wheel
 * step, adding intermediate steps between two power of two improves the user
 * experience. so that, for a number of steps defined by \f$N\f$, we define the
 * zoom value to scale value conversion function as:
 *
 * \f$scale = z2s(zoom) = 2^{zoom / N}\f$
 *
 * As it is useful to access to power of two steps and intermediate power of two
 * steps, the formula can be rewritten into:
 *
 *\f$z2s(zoom) = z2s_{int}(zoom) \times z2s_{dec}(zoom)\f$
 *
 * where:
 *
 * \f$z2s_{int}(zoom) = 2^{\lfloor zoom / N \rfloor}\f$
 *
 * and
 *
 * \f$z2s_{dec}(zoom) = 2^{zoom \bmod N}\f$
 */

template <int STEPS = 5>
class PVZoomConverterScaledPowerOfTwo : public PVZoomConverter
{
public:
	/**
	 * store the template parameter STEPS
	 */
	constexpr static int zoom_steps = STEPS;
	/**
	 * store the constants used by zoom_to_scale_decimal()
	 */
	constexpr static double root_step = pow(2.0, 1.0 / zoom_steps);

	virtual int scale_to_zoom(const qreal value) const override
	{
		// non simplified formula is: log2(1/value) / log2(root_steps)
		return floor(zoom_steps * log2(value));
	}

	virtual qreal zoom_to_scale(const int value) const override
	{
		return zoom_to_scale_integer(value) * zoom_to_scale_decimal(value);
	}

	/**
	 * Returns the integer part of the zoom_to_scale() formula
	 *
	 * @return the integer part of the zoom_to_scale() formula
	 */
	qreal zoom_to_scale_integer(const int value) const
	{
		return pow(2.0, value / zoom_steps);
	}

	/**
	 * Returns the decimal part of the zoom_to_scale() formula
	 *
	 * @return the decimal part of the zoom_to_scale() formula
	 */
	qreal zoom_to_scale_decimal(const int value) const
	{
		return pow(root_step, value % zoom_steps);
	}
};

}

#endif // PVPARALLELVIEW_PVZOOMCONVERTERSCALEDPOWEROFTWO_H
