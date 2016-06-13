/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/inendi/PVPlotted.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

// Plotted updating waxes
//

IMPL_WAX(Inendi::PVPlotted::plotting_updated, plotted, args)
{
	about_to_refresh_observers(&plotted->get_plotting());
	call_object_default<Inendi::PVPlotted, FUNC(Inendi::PVPlotted::plotting_updated)>(plotted,
	                                                                                  args);
	refresh_observers(&plotted->get_plotting());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
