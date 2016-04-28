/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_NETWORK_H
#define PVCORE_NETWORK_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/dumbnet.h>

namespace PVCore
{

struct Network
{
	static bool ipv4_aton(QString const& ip, uint32_t& ip_n);
	static bool ipv4_aton(const char* buffer, size_t n, uint32_t& ret);
	static bool ipv4_a16ton(const uint16_t* buffer, size_t n, uint32_t& ret);
	static char* ipv4_ntoa(const ip_addr_t addr);
};
}

#endif /* PVCORE_NETWORK_H */
