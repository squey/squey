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

class PVLayerFilterHeatline : public PVLayerFilter {
public:
	PVLayerFilterHeatline(PVCore::PVArgumentList const& l = PVLayerFilterHeatline::default_args());
	void operator()(PVLayer& in, PVLayer &out);
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preset() const;
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
protected:
	void post(const PVLayer &in, PVLayer &out,
	          const double ratio, const double fmin, const double fmax,
	          const PVRow line_id);

	virtual QString menu_name() const { return "Frequency gradient"; }

	CLASS_FILTER(Inendi::PVLayerFilterHeatline)
};

}

#endif
