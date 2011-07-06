// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVPlottingFilterMinmax.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	// All the plottign types that can plot vs. their min and max values are registered here
	REGISTER_FILTER("float_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_FILTER("integer_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_FILTER("string_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_FILTER("time_minmax", Picviz::PVPlottingFilterMinmax);
}
