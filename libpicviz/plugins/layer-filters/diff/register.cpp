// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterDiff.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Diff", Picviz::PVLayerFilterDiff);
}
