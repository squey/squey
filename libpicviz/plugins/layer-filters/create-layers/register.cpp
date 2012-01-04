// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include "PVLayerFilterCreateLayers.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Create Layers/Webmails", Picviz::PVLayerFilterCreateLayers);
	DECLARE_TAG(PVAXIS_TAG_DOMAIN, PVAXIS_TAG_DOMAIN_DESC, Picviz::PVLayerFilterCreateLayers);
}
