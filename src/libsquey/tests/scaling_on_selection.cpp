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

#include <squey/PVScaled.h>
#include <squey/PVScalingProperties.h>
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <iostream>

static constexpr const char* csv_file = TEST_FOLDER "/picviz/scaling_on_selection.csv";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/picviz/scaling_on_selection.csv.format";

// The column holds the values 0 to 99, one per row.
static constexpr const PVRow ROW_COUNT = 100;
static constexpr const PVRow SELECTED_FIRST = 40;
static constexpr const PVRow SELECTED_LAST = 49;

static constexpr const uint32_t AXIS_MAX = std::numeric_limits<uint32_t>::max();

/**
 * The position of a row on the axis, read the way the views read it.
 *
 * The scaled values are stored inverted, the top of the axis being zero.
 */
static uint32_t axis_position(Squey::PVScaled const& scaled, PVRow row)
{
	return ~scaled.get_value(row, PVCol(0));
}

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::Mapped);

	Squey::PVScaled& scaled = env.compute_scaling();

	PV_VALID(scaled.get_row_count(), ROW_COUNT);

	// Scaled over every row: value 0 sits at the bottom of the axis and 99 at its
	// top, so a ten-value stretch in the middle covers a tenth of it.
	const uint32_t before_first = axis_position(scaled, SELECTED_FIRST);
	const uint32_t before_last = axis_position(scaled, SELECTED_LAST);
	PV_ASSERT_VALID(before_last > before_first);
	PV_ASSERT_VALID(before_last - before_first < AXIS_MAX / 5);

	Squey::PVSelection sel(ROW_COUNT);
	sel.select_none();
	for (PVRow row = SELECTED_FIRST; row <= SELECTED_LAST; row++) {
		sel.set_line(row, true);
	}

	// Nothing asks to scale over the selection yet.
	PV_ASSERT_VALID(not scaled.update_scaling_on_selection(sel));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), before_first);

	scaled.set_scale_on_selection(true);
	PV_ASSERT_VALID(scaled.scale_on_selection(PVCol(0)));
	PV_ASSERT_VALID(scaled.update_scaling_on_selection(sel));

	// The selection now spans the axis, its ends on the ends of the axis.
	PV_VALID(axis_position(scaled, SELECTED_FIRST), (uint32_t)0);
	PV_VALID(axis_position(scaled, SELECTED_LAST), AXIS_MAX);

	// Asked again for the same rows, nothing has moved: the column already sits on
	// these bounds, and saying otherwise would cost a rebuild of every zone tree
	// it takes part in.
	PV_ASSERT_VALID(not scaled.update_scaling_on_selection(sel));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), (uint32_t)0);
	PV_VALID(axis_position(scaled, SELECTED_LAST), AXIS_MAX);

	// Order is kept within the selection, and evenly so: the ten values are a
	// ninth of the axis apart.
	for (PVRow row = SELECTED_FIRST + 1; row <= SELECTED_LAST; row++) {
		PV_ASSERT_VALID(axis_position(scaled, row) > axis_position(scaled, row - 1));
	}

	// Rows outside the selection are pinned to the ends rather than dropped.
	for (PVRow row = 0; row < SELECTED_FIRST; row++) {
		PV_VALID(axis_position(scaled, row), (uint32_t)0);
	}
	for (PVRow row = SELECTED_LAST + 1; row < ROW_COUNT; row++) {
		PV_VALID(axis_position(scaled, row), AXIS_MAX);
	}

	// An axis told to stay out of it keeps the bounds of its every row.
	scaled.get_properties_for_col(PVCol(0))
	    .set_selection_scaling(Squey::PVScalingProperties::ESelectionScaling::Disabled);
	PV_ASSERT_VALID(not scaled.scale_on_selection(PVCol(0)));
	PV_ASSERT_VALID(scaled.update_scaling_on_selection(sel));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), before_first);
	PV_VALID(axis_position(scaled, SELECTED_LAST), before_last);

	// And one told to always take part does so whatever the view is set to.
	scaled.set_scale_on_selection(false);
	scaled.get_properties_for_col(PVCol(0))
	    .set_selection_scaling(Squey::PVScalingProperties::ESelectionScaling::Enabled);
	PV_ASSERT_VALID(scaled.update_scaling_on_selection(sel));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), (uint32_t)0);
	PV_VALID(axis_position(scaled, SELECTED_LAST), AXIS_MAX);

	// Dropping the domain gives the whole column back.
	scaled.clear_selection_domain();
	PV_VALID(axis_position(scaled, SELECTED_FIRST), before_first);
	PV_VALID(axis_position(scaled, SELECTED_LAST), before_last);

	// An empty selection has no bounds to spread: refused, and nothing moves.
	Squey::PVSelection empty(ROW_COUNT);
	empty.select_none();
	PV_ASSERT_VALID(not scaled.update_scaling_on_selection(empty));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), before_first);

	// The same, driven the way a view drives it: through the selection a view
	// hands out, which is what the toolbar button passes along.
	scaled.get_properties_for_col(PVCol(0))
	    .set_selection_scaling(Squey::PVScalingProperties::ESelectionScaling::Inherit);
	scaled.set_scale_on_selection(true);

	Squey::PVView& view = scaled.emplace_add_child();
	view.set_selection_view(sel);

	const Squey::PVSelection& view_sel = view.get_real_output_selection();
	PV_VALID(view_sel.bit_count(), (size_t)(SELECTED_LAST - SELECTED_FIRST + 1));

	PV_ASSERT_VALID(scaled.update_scaling_on_selection(view_sel));
	PV_VALID(axis_position(scaled, SELECTED_FIRST), (uint32_t)0);
	PV_VALID(axis_position(scaled, SELECTED_LAST), AXIS_MAX);

	// An axis scaled by port ranges takes part too. Its ports are placed by the
	// range they belong to, as ever, and the stretch of axis the selected ones
	// ended up on is then pulled over the whole of it -- the mode says how far
	// apart they sit, selection scaling says over what.
	scaled.get_properties_for_col(PVCol(0)).set_mode("port");
	scaled.get_properties_for_col(PVCol(0))
	    .set_selection_scaling(Squey::PVScalingProperties::ESelectionScaling::Inherit);
	scaled.set_scale_on_selection(false);
	scaled.clear_selection_domain();

	const uint32_t port_first = axis_position(scaled, SELECTED_FIRST);
	const uint32_t port_last = axis_position(scaled, SELECTED_LAST);

	// Ports 40 to 49 are all system ports, so they share the bottom of the axis.
	PV_ASSERT_VALID(port_last > port_first);
	PV_ASSERT_VALID(port_last - port_first < AXIS_MAX / 5);

	scaled.set_scale_on_selection(true);
	PV_ASSERT_VALID(scaled.update_scaling_on_selection(sel));

	PV_VALID(axis_position(scaled, SELECTED_FIRST), (uint32_t)0);
	PV_VALID(axis_position(scaled, SELECTED_LAST), AXIS_MAX);
	for (PVRow row = SELECTED_FIRST + 1; row <= SELECTED_LAST; row++) {
		PV_ASSERT_VALID(axis_position(scaled, row) > axis_position(scaled, row - 1));
	}

	return 0;
}
