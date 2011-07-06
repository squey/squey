// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldSplitterPcapPacket.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("pcap", PVFilter::PVFieldSplitterPcapPacket);
}
