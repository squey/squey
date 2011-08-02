// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVPlottingFilterTimeDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("time_default", Picviz::PVPlottingFilterTimeDefault);
}
