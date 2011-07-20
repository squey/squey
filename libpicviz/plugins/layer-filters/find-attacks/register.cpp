// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVLayerFilterFindAttacks.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("Find/Attacks", Picviz::PVLayerFilterFindAttacks);
}
