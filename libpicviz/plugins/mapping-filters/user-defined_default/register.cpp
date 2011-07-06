// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVMappingFilterUserdefinedDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("user-defined_default", Picviz::PVMappingFilterUserdefinedDefault);
}
