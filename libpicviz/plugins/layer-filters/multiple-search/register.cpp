// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterMultipleSearch.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("search-multiple", Picviz::PVLayerFilterMultipleSearch);
}
