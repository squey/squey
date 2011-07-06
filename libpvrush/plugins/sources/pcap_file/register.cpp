#include <pvcore/PVClassLibrary.h>
#include "PVSourceCreatorPcapfile.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("pcap_file", PVRush::PVSourceCreatorPcapfile);
}
