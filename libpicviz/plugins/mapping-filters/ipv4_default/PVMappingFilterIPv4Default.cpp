#include "PVMappingFilterIPv4Default.h"
#include <pvcore/network.h>
#include <arpa/inet.h>

float Picviz::PVMappingFilterIPv4Default::operator()(QString const& value)
{
	uint32_t intval = 0;
	if (!PVCore::Network::ipv4_aton(value, intval)) {
		PVLOG_ERROR("ipv4_mapping: IPv4 address %s has an invalid format. Returns 0\n", qPrintable(value));
		return 0;
	}
	intval = ntohl(intval);

	return (float)intval;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIPv4Default)
