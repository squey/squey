// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVMappingFilterTimeWeek.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("time_week", Picviz::PVMappingFilterTimeWeek);
}
