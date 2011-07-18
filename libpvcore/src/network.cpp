/*
 * $Id: network.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <pvcore/network.h>

#include <QString>
#include <stdlib.h>

bool PVCore::Network::ipv4_aton(QString const& ip, uint32_t& ip_n)
{
#if 0 // This is far too slow
	struct addr addr;
	if (addr_aton(ip.toLatin1().constData(), &addr) == -1) {
		return false;
	}
	ip_n = ntohl(addr.addr_ip);
#endif

	int count = 2;

	if (ip.isEmpty()) {
		return false;
	}

	QByteArray value_ba = ip.toLatin1();
	char* buffer = (char*) value_ba.constData();
	char* buffer_org = buffer;

	buffer = strchr(buffer, '.');
	if (!buffer) {
		return false;
	}
	*buffer = 0;
	ip_n = ((uint32_t) atol(buffer_org)) << 24;
	buffer++;
	char* buffer_prev = buffer;
	while(count > 0) {
		buffer = strchr(buffer_prev, '.');
		if (!buffer) {
			return false;
		}
		*buffer = 0;
		ip_n |= ((uint32_t)atol(buffer_prev)) << count*8;
		buffer_prev = buffer+1;
		count--;
	}
	ip_n |= (uint32_t) atol(buffer_prev);

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
