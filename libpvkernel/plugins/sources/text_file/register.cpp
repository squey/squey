/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorTextfile.h"
#include "PVSourceCreatorRemoteTextfile.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("text_file", PVRush::PVSourceCreatorTextfile);
	REGISTER_CLASS("remote_text_file", PVRush::PVSourceCreatorRemoteTextfile);
}
