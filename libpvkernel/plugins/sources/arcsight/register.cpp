#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorArcsight.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("arcsight", PVRush::PVSourceCreatorArcsight);
}
