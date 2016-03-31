/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTERHeatline_H
#define INENDI_PVLAYERFILTERHeatline_H


#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

namespace Inendi {

	/**
	 * HeatLine plugins create a new layer based on frequency information.
	 *
	 * Frequency is the number of occurrence for a values compare to the number of row.
	 *
	 * Line are unselected if they are out of the frequency range.
	 * Selected line are colorized from red (max freq) to green (min freq).
	 */
class PVLayerFilterHeatline : public PVLayerFilter {

public:
	PVLayerFilterHeatline(PVCore::PVArgumentList const& l = PVLayerFilterHeatline::default_args());

	/**
	 * Process function that handle computation from in layer to out layer.
	 *
	 * It computes frequency for each values on selected axis
	 */
	void operator()(PVLayer const& in, PVLayer &out) override;

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
	void post(PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id);

	/**
	 * Name to display in the menu.
	 */
	QString menu_name() const override { return "Frequency gradient"; }

	CLASS_FILTER(Inendi::PVLayerFilterHeatline)
};

}

#endif
