/**
 * \file PVMappingFilterIPv4Default.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterIPv4Default.h"
#include <pvkernel/core/network.h>
#include <pvkernel/core/dumbnet.h>

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterIPv4Default::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	const ssize_t size = values.size();
	
#pragma omp parallel
	{
		QString stmp;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			values[i].get_qstr(stmp);
			uint32_t res = 0;
			PVCore::Network::ipv4_aton(stmp, res);
			_dest[i].storage_as_uint() = res;
		}
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIPv4Default)
