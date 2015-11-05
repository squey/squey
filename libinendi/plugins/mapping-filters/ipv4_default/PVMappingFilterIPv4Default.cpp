/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterIPv4Default.h"
#include <pvkernel/core/network.h>

Inendi::PVMappingFilter::decimal_storage_type Inendi::ipv4_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ret;
	if (!PVCore::Network::ipv4_aton(buf, size, ret)) {
		ret = 0;
	}
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

Inendi::PVMappingFilter::decimal_storage_type Inendi::ipv4_mapping::process_utf16(uint16_t const* buf, size_t size, PVMappingFilter*)
{
	Inendi::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ret;
	if (!PVCore::Network::ipv4_a16ton(buf, size, ret)) {
		ret = 0;
	}
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterIPv4Default)
