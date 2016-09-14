/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_NETWORK_H
#define PVCORE_NETWORK_H

#include <cstddef>
#include <cstdint>

namespace PVCore
{

struct Network {
	static bool ipv4_aton(const char* str, size_t n, uint32_t& ret);
};
} // namespace PVCore

#endif /* PVCORE_NETWORK_H */
