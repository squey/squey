/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "PVMappingFilterHostDefault.h"

#include <pvkernel/core/network.h>

#include <algorithm>

#include <pvkernel/core/dumbnet.h>

#include <pvcop/db/read_dict.h>

/**
 * Return a values which sort strings based on theirs 4 first chars.
 */
static uint32_t compute_str_factor(const char* str, size_t len)
{
	switch(len) {
		case 0:
			return 0;
		case 1:
			return str[0] << 24;
		case 2:
			return str[0] << 24 | str[1] << 16;
		case 3:
			return str[0] << 24 | str[1] << 16 | str[2] << 8;
		default:
			return str[0] << 24 | str[1] << 16 | str[2] << 8 | str[3];
	}
}


Inendi::PVMappingFilterHostDefault::PVMappingFilterHostDefault():
	PVMappingFilter()
{
	INIT_FILTER_NOPARAM(PVMappingFilterHostDefault);
}

/**
 * Ip values are between 0 and 2**31 while string values are between 2**31 and 2**32
 */
Inendi::PVMappingFilter::decimal_storage_type*
Inendi::PVMappingFilterHostDefault::operator()(PVCol const col, PVRush::PVNraw const& nraw) {
	auto array = nraw.collection().column(col);
	auto& core_array = array.to_core_array<uint32_t>();

	// Store mapping for each dict value.
	auto& dict = *nraw.collection().dict(col);
	std::vector<uint32_t> ret(dict.size());
	size_t i = 0;
	for(const char* c: dict) {
		if (PVCore::Network::ipv4_aton(c, strlen(c), ret[i])) {
			// That goes to the first half of the space
			// Two consecutive value may have the same mapping value.
			ret[i] >>= 1;
		}
		else {
			// Take the first four characters
			ret[i] = compute_str_factor(c, strlen(c));
			// That goes to the other half!
			ret[i] = (ret[i] >> 1) | 0x80000000;
		}
		++i;
	}

	// Copy mapping value based on computation from dict.
	for(size_t row=0; row< array.size(); row++) {
		_dest[row].storage_as_uint() = ret[core_array[row]];
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterHostDefault)
