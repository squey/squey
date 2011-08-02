// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVLayerFilterAxisGradient.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("Axis Gradient", Picviz::PVLayerFilterAxisGradient);
}
