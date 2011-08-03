#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorTexthdfs.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("text_hdfs", PVRush::PVSourceCreatorTexthdfs);
}
