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
#include <pvkernel/core/squey_assert.h>
#include <pvcop/db/algo.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/picviz/scaling_one_value.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/scaling_one_value.csv.format";

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::Mapped);

	auto const& scaled = env.compute_scaling().get_scaled(PVCol(0));

	// Check mapping is the same as NRaw value.
	PVRush::PVNraw const& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();
	const pvcop::db::array& column = nraw.column(PVCol(0));

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	Squey::PVSelection sel(column.size());
	sel.select_all();
	pvcop::db::algo::distinct(column, col1_out, col2_out, sel);

	PV_ASSERT_VALID(col1_out.size() == 1 && col2_out.size() == 1);

	bool all_same_values = std::all_of(scaled.begin() + 1, scaled.end(),
	                                   [&](const uint32_t& v) { return v == *scaled.begin(); });

	PV_ASSERT_VALID(all_same_values);

	return 0;
}
