#include <pvkernel/core/PVClassLibrary.h>
#include <pvguiqt/common.h>

#include <pvguiqt/PVDisplayViewAxesCombination.h>
#include <pvguiqt/PVDisplayViewListing.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>

void PVGuiQt::common::register_displays()
{
	REGISTER_CLASS("guiqt_axes-combination", PVDisplays::PVDisplayViewAxesCombination);
	REGISTER_CLASS("guiqt_layer-stack", PVDisplays::PVDisplayViewLayerStack);
	REGISTER_CLASS("guiqt_listing", PVDisplays::PVDisplayViewListing);
}
