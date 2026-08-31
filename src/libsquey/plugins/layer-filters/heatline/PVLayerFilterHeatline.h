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

#ifndef SQUEY_PVLAYERFILTERHeatline_H
#define SQUEY_PVLAYERFILTERHeatline_H

#include <pvbase/types.h>

#include <squey/PVLayer.h>
#include <squey/PVLayerFilter.h>

namespace Squey
{

/**
 * HeatLine plugins create a new layer based on frequency information.
 *
 * Frequency is the number of occurrence for a values compare to the number of row.
 *
 * Line are unselected if they are out of the frequency range.
 * Selected line are colorized from red (max freq) to green (min freq).
 */
class PVLayerFilterHeatline : public PVLayerFilter
{

  public:
	PVLayerFilterHeatline(PVCore::PVArgumentList const& l = PVLayerFilterHeatline::default_args());

	/**
	 * Process function that handle computation from in layer to out layer.
	 *
	 * It computes frequency for each values on selected axis
	 */
	void operator()(PVLayer const& in, PVLayer& out) override;

	/**
	 * Get preset keys.
	 *
	 * It is the same as args except for axis.
	 */
	PVCore::PVArgumentKeyList get_args_keys_for_preset() const override;

	/**
	 * Defines args.
	 *
	 * scale : scale for colorization (log or lineare).
	 * axis : Axis on which we want to apply this frequency computation.
	 * range : min and max frequency we want to handle.
	 */
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view) override;

  protected:
	/**
	 * Set color and selection parameter to the line.
	 *
	 * @param out: Changed filter
	 * @param ratio: Value in [0, 1]. 1 Means freq max, 0 mean freq min.
	 * @param fmin: Under this frequency, we don't want to select the line.
	 * @param fmax: Above this frequency, we don't want to select the line.
	 * @param line_id: The line to set these informations.
	 */
	void post(PVLayer& out,
	          const double ratio,
	          const double fmin,
	          const double fmax,
	          const PVRow line_id);

	/**
	 * Name to display in the menu.
	 */
	QString menu_name() const override { return "Frequency gradient"; }

	CLASS_FILTER(Squey::PVLayerFilterHeatline)
};
}

#endif
