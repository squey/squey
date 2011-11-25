// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterSnortForLogs.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Snort For Logs", Picviz::PVLayerFilterSnortForLogs);
}
