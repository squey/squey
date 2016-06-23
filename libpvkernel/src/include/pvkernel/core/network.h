/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_NETWORK_H
#define PVCORE_NETWORK_H

#include <QString>

namespace PVCore
{

struct Network {
	static bool ipv4_aton(const char* buffer, size_t n, uint32_t& ret);
};
}

#endif /* PVCORE_NETWORK_H */
