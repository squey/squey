// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVMappingFilterStringSort.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	REGISTER_FILTER("string_sort", Picviz::PVMappingFilterStringSort);
}
