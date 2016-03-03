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

/**
 * Return a values which sort strings based on theirs 4 first chars.
 */
static uint32_t compute_str_factor(const char* str, size_t len)
{
	uint32_t res = *reinterpret_cast<const uint32_t*>(str);
	switch(len) {
		case 0:
			return 0;
		case 1:
			return res & 0xFF000000;
		case 2:
			return res & 0xFFFF0000;
		case 3:
			return res & 0xFFFFFF00;
		default:
			return res;
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
Inendi::PVMappingFilter::decimal_storage_type Inendi::PVMappingFilterHostDefault::process_cell(const char* buf, size_t size)
{
	uint32_t ret;
	if (PVCore::Network::ipv4_aton(buf, size, ret)) {
		// That goes to the first half of the space
		// Two consecutive value may have the same mapping value.
		ret >>= 1;
	}
	else {
		// Take the first four characters
		ret = compute_str_factor(buf, size);
		// That goes to the other half!
		ret = (ret >> 1) | 0x80000000;
	}

	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterHostDefault)
