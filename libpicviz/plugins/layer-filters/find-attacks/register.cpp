// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterFindAttacks.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Find/Attacks", Picviz::PVLayerFilterFindAttacks);
	DECLARE_TAG("user-agent", Picviz::PVLayerFilterFindAttacks);
	DECLARE_TAG("url", Picviz::PVLayerFilterFindAttacks);
}
