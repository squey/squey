#ifndef PVFILTER_PVFIELDSPLITTERPCAPPACKET_FILE_H
#define PVFILTER_PVFIELDSPLITTERPCAPPACKET_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterPcapPacket : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterPcapPacket();
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

	CLASS_FILTER(PVFilter::PVFieldSplitterPcapPacket)
};

}

#endif
