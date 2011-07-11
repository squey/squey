#include "ip.h"
#include <pvcore/general.h>

#include <stdlib.h>

bool parse_ipv4(QString const& value, uint32_t& intval)
{
	char *buffer_org = NULL, *buffer;
	intval = 0;
	int count = 2;

	if (value.isEmpty()) {
		return false;
	}

	buffer_org = strdup(value.toLatin1().data());
	buffer = buffer_org;

	buffer = strchr(buffer, '.');
	if (!buffer) {
		return false;
	}
	*buffer = 0;
	intval = ((uint32_t) atoi(buffer_org)) << 24;
	buffer++;
	char* buffer_prev = buffer;
	while(count > 0) {
		buffer = strchr(buffer_prev, '.');
		if (!buffer) {
			return false;
		}
		*buffer = 0;
		intval |= ((uint32_t)atoi(buffer_prev)) << count*8;
		buffer_prev = buffer+1;
		count--;
	}
	intval |= (uint32_t) atoi(buffer_prev);

	free(buffer_org);
	return true;
}
