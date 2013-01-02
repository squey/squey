// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeArcsight.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("arcsight", PVRush::PVInputTypeArcsight);
}
