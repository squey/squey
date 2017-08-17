/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterPort.h"

using plotting_t = Inendi::PVPlottingFilterPort::value_type;
using port_plotting_t = Inendi::PVPlottingFilterPort::port_plotting_t;

static void compute_port_plotting(pvcop::db::array const& mapped,
                                  const pvcop::db::selection& invalid_selection,
                                  pvcop::core::array<plotting_t>& dest)
{
	auto& values = mapped.to_core_array<port_plotting_t>();

	const double invalid_range =
	    invalid_selection ? Inendi::PVPlottingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const plotting_t max_plot_value = std::numeric_limits<plotting_t>::max();
	const plotting_t threshold1 = (plotting_t)(0.3 * max_plot_value);
	const plotting_t threshold2 = (plotting_t)(0.6 * max_plot_value);

	const plotting_t valid_offset = (plotting_t)(max_plot_value * invalid_range);

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		const port_plotting_t v = values[i];
		bool invalid = (invalid_selection and invalid_selection[i]);
		double delta = 0;

		plotting_t x_min = 0, x_max = 0, y_min = 0, y_max = 0;

		if (invalid) {
			dest[i] = ~plotting_t(0);
			continue;
		} else if (v < 1024) {

			// Distribute ports linearly in [valid_offset, alpha1*Max-1]
			x_min = 0;
			x_max = 1023;
			y_min = valid_offset;
			y_max = threshold1 - 1;

		} else if (v >= 1024 && v <= 49151) {
			// Distribute ports linearly in [alpha1*Max, alpha2*Max-1]
			x_min = 1024;
			x_max = 49151;
			y_min = threshold1;
			y_max = threshold2 - 1;

		} else {

			// Distribute ports linearly in [alpha2*Max, Max]
			x_min = 49152;
			x_max = (plotting_t)(1UL << 16);
			y_min = threshold2;
			y_max = max_plot_value;
		}
		delta = (y_max - y_min) / (x_max - x_min);
		dest[i] = ~plotting_t(((v - x_min) * delta) + y_min);
	}
}

void Inendi::PVPlottingFilterPort::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              const pvcop::db::selection& invalid_selection,
                                              pvcop::core::array<plotting_t>& dest)
{
	assert(dest);
	// assert(mapped.type() == "number_uint16");

	compute_port_plotting(mapped, invalid_selection, dest);
}
