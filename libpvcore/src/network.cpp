/*
 * $Id: network.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QString>

#include <pvcore/network.h>

bool PVCore::Network::ipv4_aton(QString const& ip, uint32_t& ip_n)
{
	struct in_addr addr;
	if (inet_aton(ip.toLatin1().constData(), &addr) == 0) {
		return false;
	}
	ip_n = addr.s_addr;
	return true;
}

char* PVCore::Network::ipv4_ntoa(const ip_addr_t addr)
{
	struct in_addr addr_source;

#ifdef WIN32
	addr_source.s_addr = (unsigned long)addr;
#else
	addr_source.s_addr = (in_addr_t)addr;
#endif
	return inet_ntoa(addr_source);
}


#if 0
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
#endif
