// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldFilterRegexpGrep.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("regexp", PVFilter::PVFieldFilterRegexpGrep);
	//REGISTER_FILTER("filter_regexp", PVFilter::PVFieldFilterRegexpGrep);
}
