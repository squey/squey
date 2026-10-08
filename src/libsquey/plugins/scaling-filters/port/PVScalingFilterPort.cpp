//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVScalingFilterPort.h"

#include <algorithm>

using scaling_t = Squey::PVScalingFilterPort::value_type;
using port_scaling_t = Squey::PVScalingFilterPort::port_scaling_t;

/**
 * Where a port sits on the axis: within the range it belongs to -- system,
 * registered, dynamic -- each of which is given a fixed share of the axis.
 */
static scaling_t port_position(port_scaling_t v, scaling_t valid_offset)
{
	const scaling_t max_plot_value = std::numeric_limits<scaling_t>::max();
	const auto threshold1 = (scaling_t)(0.3 * max_plot_value);
	const auto threshold2 = (scaling_t)(0.6 * max_plot_value);

	scaling_t x_min = 0, x_max = 0, y_min = 0, y_max = 0;

	if (v < 1024) {
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
		x_max = ((scaling_t)1 << 16);
		y_min = threshold2;
		y_max = max_plot_value;
	}

	const double delta = (y_max - y_min) / (x_max - x_min);
	return (scaling_t)(((v - x_min) * delta) + y_min);
}

static void compute_port_scaling(pvcop::db::array const& mapped,
                                  const pvcop::db::selection& invalid_selection,
                                  const pvcop::db::selection& domain_selection,
                                  pvcop::core::array<scaling_t>& dest)
{
	auto& values = mapped.to_core_array<port_scaling_t>();

	const double invalid_range =
	    invalid_selection ? Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const scaling_t max_plot_value = std::numeric_limits<scaling_t>::max();

	const auto valid_offset = (scaling_t)(max_plot_value * invalid_range);

	// Under selection scaling the ports are placed as ever -- their ranges are
	// what this filter is for -- and the stretch of axis the selected ones ended
	// up on is then pulled over the whole of it. Ports keep their order and the
	// gaps between them, ranges included; what a narrow selection no longer keeps
	// is the share of the axis its range was given, which is the point.
	bool spread = false;
	scaling_t selected_low = 0;
	scaling_t selected_high = 0;

	if (domain_selection) {
		for (size_t i = 0; i < values.size(); i++) {
			if (not domain_selection[i] or (invalid_selection and invalid_selection[i])) {
				continue;
			}
			const scaling_t position = port_position(values[i], valid_offset);
			if (not spread) {
				selected_low = position;
				selected_high = position;
				spread = true;
			} else {
				selected_low = std::min(selected_low, position);
				selected_high = std::max(selected_high, position);
			}
		}
	}

	// A selection sitting on one port has no stretch to pull: it would send every
	// row to the same place, so the ports keep the axis they already had.
	const bool stretch = spread and selected_high > selected_low;
	const double ratio =
	    stretch ? (max_plot_value - valid_offset) / (double)(selected_high - selected_low) : 0.;

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		if (invalid_selection and invalid_selection[i]) {
			dest[i] = ~scaling_t(0);
			continue;
		}

		const scaling_t position = port_position(values[i], valid_offset);

		dest[i] = ~(stretch ? Squey::PVScalingFilter::clamp_to_axis(
		                          ((double)position - (double)selected_low) * ratio + valid_offset,
		                          valid_offset)
		                    : position);
	}
}

void Squey::PVScalingFilterPort::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              const pvcop::db::selection& invalid_selection,
                                              const pvcop::db::selection& domain_selection,
                                              pvcop::core::array<scaling_t>& dest)
{
	assert(dest);

	compute_port_scaling(mapped, invalid_selection, domain_selection, dest);
}
