/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterPort.h"

using plotting_t = Inendi::PVPlottingFilter::value_type;

template <class T>
static void compute_port_plotting(pvcop::db::array const& mapped,
                                  const pvcop::db::selection& invalid_selection,
                                  pvcop::core::array<plotting_t>& dest)
{
	auto& values = mapped.to_core_array<T>();

	const double invalid_range =
	    invalid_selection ? Inendi::PVPlottingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const size_t valid_offset = std::numeric_limits<plotting_t>::max() * invalid_range;
	const double ratio = 1 - invalid_range;

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		const T v = values[i];
		bool invalid = (invalid_selection and invalid_selection[i]);

		if (invalid) {
			dest[i] = ~plotting_t(0);
		} else if (v <= 1024) {
			// Save value on : 0b0xxxxxxxxx0000000000000000000000
			// Move "upper" (31 - sizeof(1024 - 1)) minus 1 to give the first bit for upper port
			dest[i] = ~plotting_t(((v << (31 - 9 - 1)) * ratio) + valid_offset);
		} else {
			// Save value on : 0b1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
			// Reparted upper port in "minmax" on 31 bits and set the 32th to mark distinguish upper
			// port from lower port
			dest[i] =
			    ~plotting_t(((((plotting_t)(((uint64_t)(v - (1UL << 10)) * (uint64_t)(1UL << 31)) /
			                                (uint64_t)(((1UL << 16) - 1) - (1UL << 10))) -
			                   1) |
			                  0x80000000UL) *
			                 ratio) +
			                valid_offset);
		}
	}
}

void Inendi::PVPlottingFilterPort::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              const pvcop::db::selection& invalid_selection,
                                              pvcop::core::array<plotting_t>& dest)
{
	assert(dest);

	if (mapped.type() == pvcop::db::type_uint32) {
		compute_port_plotting<uint32_t>(mapped, invalid_selection, dest);
	} else {
		assert(mapped.type() == pvcop::db::type_int32);
		compute_port_plotting<int32_t>(mapped, invalid_selection, dest);
	}
}
