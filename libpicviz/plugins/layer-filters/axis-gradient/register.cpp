// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterAxisGradient.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Axis Gradient", Picviz::PVLayerFilterAxisGradient);
}
