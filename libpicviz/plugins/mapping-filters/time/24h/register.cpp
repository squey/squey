// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterTime24h.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("time_24h", Picviz::PVMappingFilterTime24h);
}
