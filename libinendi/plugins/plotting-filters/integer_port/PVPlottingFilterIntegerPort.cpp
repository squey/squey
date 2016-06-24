/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterIntegerPort.h"

uint32_t* Inendi::PVPlottingFilterIntegerPort::operator()(pvcop::db::array const& mapped)
{
	assert(_dest);

	auto& values = mapped.to_core_array<uint32_t>();

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		const uint32_t v = values[i];
		if (v < 1024) {
			_dest[i] = ~(v << 21);
		} else {
			_dest[i] = ~(((uint32_t)(((uint64_t)(v - 1024) * (uint64_t)(UINT_MAX)) /
			                         (uint64_t)(65535 - 1024))) |
			             0x80000000UL);
		}
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterIntegerPort)
