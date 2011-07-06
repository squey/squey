// Register the plugin in PVFilterLibrary
//

#include <pvcore/PVClassLibrary.h>
#include <pvrush/PVInputType.h>

#include "PVInputTypeHDFS.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("hdfs", PVRush::PVInputTypeHDFS);
}
