// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVMappingFilterStringDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("string_default", Picviz::PVMappingFilterStringDefault);
}
