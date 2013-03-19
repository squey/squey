
#ifndef PVPARALLELVIEW_PVAXISZOOM_H
#define PVPARALLELVIEW_PVAXISZOOM_H

#include <pvkernel/PVAlgorithms.h>
#include <pvparallelview/PVZoomConverter.h>

namespace PVParallelView
{

/**
 * @class PVAxisZoom
 *
 * This class group together all information about zoom along one axis
 */
class PVAxisZoom
{
public:
	/**
	 * CTOR :-p
	 *
	 * @param vdefault [in] the default value, its used to initialize the current value too
	 * @param vmin [in] the inclusive lower bound of the current value
	 * @param vmax [in] the inclusive uper bound of the current value
	 * @param zoom_converter [in] the zoom converter to associate to this PVAxisZoom
	 *
	 * @attention @a zoom_converter's deletion is not PVAxisZoom's job.
	 */
	PVAxisZoom(int vdefault, int vmin, int vmax, const PVZoomConverter *zoom_converter) :
		_value(vdefault),
		_value_min(vmin),
		_value_max(vmax),
		_value_default(vdefault),
		_zoom_converter(zoom_converter)
	{}

	/**
	 * Change the stored value.
	 *
	 * @param value [in] the new value
	 */
	void set_value(const int value) { _value = PVCore::clamp(value, _value_min, _value_max); }

	/**
	 * Returns the current value.
	 *
	 * @return the current value
	 */
	int get_value() const { return _value; }

	/**
	 * Resets the current value to its default value
	 */
	void reset_to_default() { _value = _value_default; }

	/**
	 * Returns the lower bound.
	 *
	 * @return the lower bound
	 */
	int get_min() const { return _value_min; }

	/**
	 * Returns the upper bound.
	 *
	 * @return the upper bound
	 */
	int get_max() const { return _value_max; }

	/**
	 * Returns the default value.
	 *
	 * @return the default value
	 */
	int get_default() const { return _value_default; }

	/**
	 * Returns the zoom converter associated with this PVAxisZoom.
	 *
	 * @return the associated zoom converter
	 */
	PVZoomConverter *get_zoom_converter() const { return _zoom_converter; }

private:
	int              _value;
	int              _value_min;
	int              _value_max;
	int              _value_default;
	PVZoomConverter *_zoom_converter;
};


}

#endif // PVPARALLELVIEW_PVAXISZOOM_H
