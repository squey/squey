// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVFieldSplitterPcapPacket.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("pcap", PVFilter::PVFieldSplitterPcapPacket);
	REGISTER_FILTER_AS("splitter_pcap", PVFilter::PVFieldSplitterPcapPacket, PVFilter::PVFieldsFilterReg);
}
