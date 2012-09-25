/**
 * \file network.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/network.h>

#include <QString>
#include <stdlib.h>

static uint32_t powui(uint32_t base, uint32_t n)
{
	uint32_t ret = 1;
	for (uint32_t i = 0; i < n; i++) {
		ret *= base;
	}   
	return ret;
}

bool PVCore::Network::ipv4_aton(QString const& ip, uint32_t& ip_n)
{
	return ipv4_a16ton((const uint16_t*) ip.constData(), ip.size(), ip_n);
}

bool PVCore::Network::ipv4_a16ton(const uint16_t* str, size_t n, uint32_t& ret)
{
	if ((n > 15) || (n == 0)) {
		return false;
	}   

	// TODO: vectorize this
	uint32_t ndots = 0;
	uint32_t nip = 0;
	int32_t cur_d = 2;
	size_t i = 0;
	while ((i < n) && isspace(str[i] & 0x00FF)) {
		i++;
	}
	char prev_c = 0;
	ret = 0;
	for (; i < n; i++) {
		const uint16_t c16 = str[i];
		if (c16 > 128) {
			return false;
		}
		const char c = (char) (c16 & 0x00FF);
		if (c == '.') {
			if (prev_c == '.') {
				return false;
			}   
			ndots++;
			nip /= powui(10, cur_d+1);
			if (nip > 255 || ndots > 3) {
				return false;
			}   
			ret |= nip << ((4-ndots)*8);
			nip = 0;
			cur_d = 2;
		}   
		else {
			if ((c < '0') || (c > '9')) {
				if (prev_c == '.') {
					return false;
				}   
				break;
			}   
			if (cur_d < 0) {
				return false;
			}   
			nip += ((uint32_t)((c-'0')))*powui(10, cur_d);
			cur_d--;
		}   
		prev_c = c;
	}   
	nip /= powui(10, cur_d+1);
	if (nip > 255) {
		return false;
	}   
	ret |= nip;
	// Check trailing characters
	for (; i < n; i++) {
		if (!isspace(str[i] & 0x00FF)) {
			return false;
		}   
	}   
	return ndots == 3;
}

bool PVCore::Network::ipv4_aton(const char* str, size_t n, uint32_t& ret)
{
	if ((n > 15) || (n == 0)) {
		return false;
	}   

	// TODO: vectorize this
	uint32_t ndots = 0;
	uint32_t nip = 0;
	int32_t cur_d = 2;
	size_t i = 0;
	while ((i < n) && isspace(str[i])) {
		i++;
	}   
	char prev_c = 0;
	ret = 0;
	for (; i < n; i++) {
		const char c = str[i];
		if (c == '.') {
			if (prev_c == '.') {
				return false;
			}   
			ndots++;
			nip /= powui(10, cur_d+1);
			if (nip > 255 || ndots > 3) {
				return false;
			}   
			ret |= nip << ((4-ndots)*8);
			nip = 0;
			cur_d = 2;
		}   
		else {
			if ((c < '0') || (c > '9')) {
				if (prev_c == '.') {
					return false;
				}   
				break;
			}   
			if (cur_d < 0) {
				return false;
			}   
			nip += ((uint32_t)((c-'0')))*powui(10, cur_d);
			cur_d--;
		}   
		prev_c = c;
	}   
	nip /= powui(10, cur_d+1);
	if (nip > 255) {
		return false;
	}   
	ret |= nip;
	// Check trailing characters
	for (; i < n; i++) {
		if (!isspace(str[i])) {
			return false;
		}   
	}   
	return ndots == 3;
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
