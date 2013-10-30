/**
 * \file PVPlotted.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVPlotted.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

// Plotted updating waxes
//

IMPL_WAX(Picviz::PVPlotted::process_parent_mapped, plotted, args)
{
	about_to_refresh_observers(&plotted->get_plotting());
	call_object_default<Picviz::PVPlotted, FUNC(Picviz::PVPlotted::process_parent_mapped)>(plotted, args);
	refresh_observers(&plotted->get_plotting());
}

IMPL_WAX(Picviz::PVPlotted::process_from_parent_mapped, plotted, args)
{
	about_to_refresh_observers(&plotted->get_plotting());
	call_object_default<Picviz::PVPlotted, FUNC(Picviz::PVPlotted::process_from_parent_mapped)>(plotted, args);
	refresh_observers(&plotted->get_plotting());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
