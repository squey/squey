#include "PVMappingFilterIPv4Default.h"


float Picviz::PVMappingFilterIPv4Default::operator()(QString const& value)
{
	char *buffer_org = NULL, *buffer;
	uint32_t intval = 0;
	int count = 2;

	if (value.isEmpty()) {
		PVLOG_ERROR("%s: Cannot get the string value! Returns 0\n", __FUNCTION__);
		return 0;
	}

	buffer_org = strdup(value.toLatin1().data());
	buffer = buffer_org;

	buffer = strchr(buffer, '.');
	if (!buffer) {
		PVLOG_ERROR("%s: ipv4_mapping: IPv4 address %s has an invalid format. Returns 0\n", __FUNCTION__, qPrintable(value));
		return 0;
	}
	*buffer = 0;
	intval = ((uint32_t) atoi(buffer_org)) << 24;
	buffer++;
	char* buffer_prev = buffer;
	while(count > 0) {
		buffer = strchr(buffer_prev, '.');
		if (!buffer) {
			PVLOG_ERROR("%s: ipv4_mapping: IPv4 address %s has an invalid format. Returns 0\n", __FUNCTION__, qPrintable(value));
			return 0;
		}
		*buffer = 0;
		intval |= ((uint32_t)atoi(buffer_prev)) << count*8;
		buffer_prev = buffer+1;
		count--;
	}
	intval |= (uint32_t) atoi(buffer_prev);

	free(buffer_org);

	return (float)intval;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIPv4Default)
