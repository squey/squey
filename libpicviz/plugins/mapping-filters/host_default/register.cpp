// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterHostDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("host_default", Picviz::PVMappingFilterHostDefault);
}
