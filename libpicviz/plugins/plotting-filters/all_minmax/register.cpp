// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterMinmax.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plottign types that can plot vs. their min and max values are registered here
	REGISTER_CLASS("float_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_CLASS("integer_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_CLASS("string_minmax", Picviz::PVPlottingFilterMinmax);
	REGISTER_CLASS("time_minmax", Picviz::PVPlottingFilterMinmax);
}
