// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterURL.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("url", PVFilter::PVFieldSplitterURL);
	REGISTER_CLASS_AS("splitter_url", PVFilter::PVFieldSplitterURL, PVFilter::PVFieldsFilterReg);
}
