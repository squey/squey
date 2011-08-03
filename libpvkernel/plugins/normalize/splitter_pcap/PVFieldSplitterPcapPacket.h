#ifndef PVFILTER_PVFIELDSPLITTERPCAPPACKET_FILE_H
#define PVFILTER_PVFIELDSPLITTERPCAPPACKET_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterPcapPacket : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterPcapPacket(PVCore::PVArgumentList const& args = PVFieldSplitterPcapPacket::default_args());
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
private:
	int _datalink_type;

	CLASS_FILTER(PVFilter::PVFieldSplitterPcapPacket)
};

}

#endif
