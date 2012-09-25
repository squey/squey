/**
 * \file PVMappingFilterIPv4Default.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterIPv4Default.h"
#include <pvkernel/core/network.h>

Picviz::PVMappingFilter::decimal_storage_type Picviz::ipv4_mapping::process_utf8(const char* buf, size_t size, PVMappingFilter*)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ret;
	if (!PVCore::Network::ipv4_aton(buf, size, ret)) {
		ret = 0;
	}
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

Picviz::PVMappingFilter::decimal_storage_type Picviz::ipv4_mapping::process_utf16(uint16_t const* buf, size_t size, PVMappingFilter*)
{
	Picviz::PVMappingFilter::decimal_storage_type ret_ds;
	uint32_t ret;
	if (!PVCore::Network::ipv4_a16ton(buf, size, ret)) {
		ret = 0;
	}
	ret_ds.storage_as_uint() = ret;
	return ret_ds;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIPv4Default)
