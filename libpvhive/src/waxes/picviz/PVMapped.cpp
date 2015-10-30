/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVMapped.h>
#include <pvhive/waxes/picviz/PVPlotted.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

// Mapped updating waxes
//

IMPL_WAX(Picviz::PVMapped::process_from_parent_source, mapped, args)
{
	for (auto const& c: mapped->get_children<Picviz::PVPlotted>()) {
		about_to_refresh_observers(&c->get_plotting());
	}
	call_object_default<Picviz::PVMapped, FUNC(Picviz::PVMapped::process_from_parent_source)>(mapped, args);
	for (auto const& c: mapped->get_children<Picviz::PVPlotted>()) {
		refresh_observers(&c->get_plotting());
	}
}

PVHIVE_CALL_OBJECT_BLOCK_END()
