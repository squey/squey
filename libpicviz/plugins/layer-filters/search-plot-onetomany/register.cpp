// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterSearchPlotOneToMany.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Plot Search/One to many", Picviz::PVLayerFilterSearchPlotOneToMany);
}
