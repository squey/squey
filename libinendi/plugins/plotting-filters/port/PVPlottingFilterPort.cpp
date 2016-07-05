/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterPort.h"

template <class T>
static void compute_port_plotting(pvcop::db::array const& mapped, uint32_t* dest)
{
	auto& values = mapped.to_core_array<T>();

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		const T v = values[i];
		if (v < 1024) {
			// Save value on : 0b0xxxxxxxxx0000000000000000000000
			// Move "upper" (31 - sizeof(1024 - 1)) minus 1 to give the first bit for upper port
			dest[i] = ~uint32_t(v << (31 - 9 - 1));
		} else {
			// Save value on : 0b1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
			// Reparted upper port in "minmax" on 31 bits and set the 32th to mark distinguish upper
			// port from lower port
			dest[i] = ~uint32_t(((uint32_t)(((uint64_t)(v - 1024) * (uint64_t)(1UL << 31)) /
			                                (uint64_t)(65535 - 1024))) |
			                    0x80000000UL);
		}
	}
}

void Inendi::PVPlottingFilterPort::
operator()(pvcop::db::array const& mapped, pvcop::db::array const&, uint32_t* dest)
{
	// FIXME : We may inform user if minmax is not in 65535 - 0
	assert(dest);

	if (mapped.type() == pvcop::db::type_uint32) {
		compute_port_plotting<uint32_t>(mapped, dest);
	} else {
		assert(mapped.type() == pvcop::db::type_int32);
		compute_port_plotting<int32_t>(mapped, dest);
	}
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterPort)
