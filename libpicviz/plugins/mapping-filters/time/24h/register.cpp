// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVMappingFilterTime24h.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("time_24h", Picviz::PVMappingFilterTime24h);
}
