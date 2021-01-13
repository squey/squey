/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvguiqt/common.h>

#include <pvguiqt/PVDisplaySourceDataTree.h>
#include <pvguiqt/PVDisplayViewAxesCombination.h>
#include <pvguiqt/PVDisplayViewListing.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>
#include <pvguiqt/PVDisplayViewPythonConsole.h>

void PVGuiQt::common::register_displays()
{
	REGISTER_CLASS("guiqt_axes-combination", PVDisplays::PVDisplayViewAxesCombination);
	REGISTER_CLASS("guiqt_datatree", PVDisplays::PVDisplaySourceDataTree);
	REGISTER_CLASS("guiqt_layer-stack", PVDisplays::PVDisplayViewLayerStack);
	REGISTER_CLASS("guiqt_listing", PVDisplays::PVDisplayViewListing);
	REGISTER_CLASS("guiqt_pythonconsole", PVDisplays::PVDisplayViewPythonConsole);
}
