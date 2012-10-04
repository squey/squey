/**
 * \file PVPlottingFilterIntegerPort.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterIntegerPort.h"


uint32_t* Picviz::PVPlottingFilterIntegerPort::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	const ssize_t size = _dest_size;

	uint32_t const* const vint = &values->storage_as_uint();
	
#pragma omp parallel for
	for (ssize_t i = 0; i < size; i++) {
		const uint32_t v = vint[i];
		if (v < 1024) {
			_dest[i] = ~(v<<21);
		} else {
			_dest[i] = ~(((uint32_t) (((uint64_t)(v-1024)*(uint64_t)(UINT_MAX))/(uint64_t)(65535-1024))) | 0x80000000UL);
		}
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterIntegerPort)
