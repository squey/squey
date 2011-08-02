// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVMappingFilterString4Bsort.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("string_4Bsort", Picviz::PVMappingFilterString4Bsort);
}
