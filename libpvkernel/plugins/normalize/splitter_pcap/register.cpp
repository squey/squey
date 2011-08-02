// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterPcapPacket.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("pcap", PVFilter::PVFieldSplitterPcapPacket);
	REGISTER_CLASS_AS("splitter_pcap", PVFilter::PVFieldSplitterPcapPacket, PVFilter::PVFieldsFilterReg);
}
