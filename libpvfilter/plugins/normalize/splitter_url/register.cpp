// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldSplitterURL.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("url", PVFilter::PVFieldSplitterURL);
}
