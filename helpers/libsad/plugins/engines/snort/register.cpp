#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "SadPluginSnortEngine.h"

LibCPPExport void register_class()
{
	REGISTER_CLASS("snort", Sad::SnortEngine);
}

