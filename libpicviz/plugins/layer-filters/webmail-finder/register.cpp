// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVLayerFilterWebmailFinder.h"
#include "../include/tags.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Find/Webmails", Picviz::PVLayerFilterWebmailFinder);
	DECLARE_TAG(PVLAYERFILTER_TAG_DOMAIN, PVLAYERFILTER_TAG_DOMAIN_DESC, Picviz::PVLayerFilterWebmailFinder);
}
