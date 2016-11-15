/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */
/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotted.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvcop/db/algo.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/picviz/plotting_one_value.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/plotting_one_value.csv.format";

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::Mapped);

	auto const& plotted = env.compute_plotting().get_plotted(0);

	// Check mapping is the same as NRaw value.
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(0);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	Inendi::PVSelection sel(column.size());
	sel.select_all();
	pvcop::db::algo::distinct(column, col1_out, col2_out, sel);

	PV_ASSERT_VALID(col1_out.size() == 1 && col2_out.size() == 1);

	bool all_same_values = std::all_of(plotted.begin() + 1, plotted.end(),
	                                   [&](const uint32_t& v) { return v == *plotted.begin(); });

	PV_ASSERT_VALID(all_same_values);

	return 0;
}
