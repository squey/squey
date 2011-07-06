// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVPlottingFilterNoprocess.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	// All the plotting types that do not process anyhting are registered here
	REGISTER_FILTER("enum_default", Picviz::PVPlottingFilterNoprocess);
	REGISTER_FILTER("string_default", Picviz::PVPlottingFilterNoprocess);
	REGISTER_FILTER("user-defined_default", Picviz::PVPlottingFilterNoprocess);
}
