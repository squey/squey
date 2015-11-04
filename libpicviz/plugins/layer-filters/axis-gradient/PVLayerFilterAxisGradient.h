/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PICVIZ_PVLAYERFILTERAXISGRADIENT_H
#define PICVIZ_PVLAYERFILTERAXISGRADIENT_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterAxisGradient
 */
class PVLayerFilterAxisGradient : public PVLayerFilter {
public:
	PVLayerFilterAxisGradient(PVCore::PVArgumentList const& l = PVLayerFilterAxisGradient::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preset() const;
	virtual QString status_bar_description();
	virtual QString detailed_description();
	virtual QString menu_name() const { return "Axis gradient"; }

public:
	static PVCore::PVArgumentList gradient_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);

	CLASS_FILTER(Picviz::PVLayerFilterAxisGradient)

};
}

#endif
