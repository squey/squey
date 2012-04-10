// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterFindSameValues.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("find-same-values", Picviz::PVLayerFilterFindSameValues);
}
