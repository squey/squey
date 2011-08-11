#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorDatabase.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("database", PVRush::PVSourceCreatorDatabase);
}
