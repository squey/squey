// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterIntegerDefault.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("integer_default", Picviz::PVMappingFilterIntegerDefault);
}
