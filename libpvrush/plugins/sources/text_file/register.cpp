#include <pvcore/PVClassLibrary.h>
#include "PVSourceCreatorTextfile.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("text_file", PVRush::PVSourceCreatorTextfile);
}
