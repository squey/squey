// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldFilterRegexpGrep.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("grep_regexp", PVFilter::PVFieldFilterRegexpGrep);
}
