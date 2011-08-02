// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterEnumDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("enum_default", Picviz::PVMappingFilterEnumDefault);
}
