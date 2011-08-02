// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVLayerFilterWebmailFinder.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("Find/Webmails", Picviz::PVLayerFilterWebmailFinder);
}
