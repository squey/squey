/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVAXISZOOM_H
#define PVPARALLELVIEW_PVAXISZOOM_H

#include <pvkernel/core/PVAlgorithms.h>
#include <pvparallelview/PVZoomConverter.h>

#include <climits>

namespace PVParallelView
{

class PVZoomableDrawingAreaConstraints;

/**
 * @class PVAxisZoom
 *
 * This class group together all information about zoom along one axis
 */
class PVAxisZoom
{
	friend class PVZoomableDrawingAreaConstraints;

  public:
	/**
	 * CTOR
	 *
	 * you have to explicitely call setters to intialize it.
	 *
	 * @attention @a zoom_converter's deletion is not PVAxisZoom's job.

	 */
	PVAxisZoom()
	    : _value(0)
	    , _value_min(INT_MIN)
	    , _value_max(INT_MAX)
	    , _value_default(0)
	    , _inverted(false)
	    , _zoom_converter(nullptr)
	{
	}

	/**
	 * Returns the current value.
	 *
	 * @return the current value
	 */
	int get_value() const { return _value; }

	/**
	 * Returns the current value restricted in its range.
	 *
	 * @return the restricted value
	 */
	int get_clamped_value() const { return PVCore::clamp(_value, _value_min, _value_max); }

	/**
	 * Returns the value relative to the lower bound
	 *
	 * @return the relative value
	 */
	int get_relative_value() const { return _value - _value_min; }

	/**
	 * Returns the clamped value relative to the lower bound.
	 *
	 * @return the relative value
	 */
	int get_clamped_relative_value() const
	{
		return PVCore::clamp(_value, _value_min, _value_max) - _value_min;
	}

	/**
	 * set range
	 *
	 * @param vmin the lower bound
	 * @param vmax the upper bound
	 */
	void set_range(int vmin, int vmax)
	{
		_value_min = vmin;
		_value_max = vmax;
	}

	/**
	 * set default value
	 *
	 * @param vdefault the default value
	 */
	void set_default_value(int vdefault) { _value_default = vdefault; }

	/**
	 * set associated PVZoomConverter
	 *
	 * @param vdefault the default value
	 */
	void set_zoom_converter(PVZoomConverter* zoom_converter) { _zoom_converter = zoom_converter; }

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
	const PVZoomConverter* get_zoom_converter() const { return _zoom_converter; }

	/**
	 * Set whether the axis orientation should be inverted.
	 */
	void set_inverted(bool inverted) { _inverted = inverted; }

	/**
	 * Returns whether the axis orientation will be inverted.
	 */
	bool inverted() const { return _inverted; }

	/**
	 * Returns true iif the zoom_converter object has been set
	 */
	bool valid() const { return _zoom_converter != nullptr; }

  protected:
	/**
	 * Change the stored value.
	 *
	 * @param value [in] the new value
	 */
	void set_value(const int value) { _value = value; }

	/**
	 * Change the stored value.
	 *
	 * The value is clamped before it is set.
	 *
	 * @param value [in] the new value
	 */
	void set_clamped_value(const int value)
	{
		_value = PVCore::clamp(value, _value_min, _value_max);
	}

	/**
	 * Resets the current value to its default value
	 */
	void reset_to_default() { _value = _value_default; }

  private:
	int _value;
	int _value_min;
	int _value_max;
	int _value_default;
	bool _inverted;
	const PVZoomConverter* _zoom_converter;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVAXISZOOM_H
