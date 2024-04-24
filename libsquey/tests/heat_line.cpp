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

#include "common.h"

#include <squey/PVLayerFilter.h>
#include <squey/PVLayer.h>

#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <pvkernel/core/squey_assert.h>

#include <chrono>
#include <fstream>
#include <string>
#include <random>

#ifndef SQUEY_BENCH
constexpr size_t CHECK_COUNT = 10000;
constexpr size_t ndup = 1;
#else
constexpr size_t ndup = 20;
#endif

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

/**
 * Tested file have 4x 0, 3x < 12499, 2x 12499, 1x >12499.
 * We can deduce color and selection from this construction with selection on
 * 60 % to 80 %.
 */
void check_line_validity(Squey::PVLayer const& out, size_t line)
{

	// Check selection
	if (line >= 12499 || line == 0) {
		PV_VALID(out.get_selection().get_line(line), false);
	} else {
		PV_VALID(out.get_selection().get_line(line), true);
	}

	if (line == 0) {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 126);
	} else if (line == 12499) {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 68);
	} else if (line < 12500) {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 101);
	} else {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 10);
	}
}

int main()
{
	// Init nraw
	pvtest::TestEnv env(filename, fileformat, ndup, pvtest::ProcessUntil::View);

	Squey::PVView* view = env.root.current_view();

	// Load every layer filter.
	Squey::common::load_layer_filters();

	// Load heat line plugin
	Squey::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name("frequency-gradient");
	Squey::PVLayerFilter::p_type fclone = filter_org->clone<Squey::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter("frequency-gradient");

	// Setup parameters.
	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(PVCol(1)));
	auto scale = args["scale"].value<PVCore::PVEnumType>();
	scale.set_sel(1);
	args["scale"].setValue(scale);
	args["colors"].setValue(PVCore::PVPercentRangeType(0.6, 0.8));

	Squey::PVLayer out("Out", view->get_row_count());
	Squey::PVLayer& in = view->get_layer_stack_output_layer();

	fclone->set_view(view);
	fclone->set_output(&out);
	fclone->set_args(args);

	auto start = std::chrono::system_clock::now();

	// Run heat line computation
	fclone->operator()(in);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef SQUEY_BENCH
	// Check result
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();
	pvcop::db::array const& col = nraw.column(PVCol(1));

	check_line_validity(out, 0);
	check_line_validity(out, 1);
	check_line_validity(out, 12499);
	check_line_validity(out, 12500);

	auto const& v = col.to_core_array<uint32_t>();
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, col.size() - 1);

	for (size_t i = 0; i < CHECK_COUNT; i++) {
		uint8_t to_check = v[distribution(generator)];
		check_line_validity(out, to_check);
	}

	// Check that we don't crash if the selection is empty
	Squey::PVLayer empty_layer("empty_layer", view->get_row_count());
	Squey::PVSelection empty_sel(view->get_row_count());
	empty_sel.select_none();
	empty_layer.get_selection() = empty_sel;
	fclone->operator()(empty_layer);
#endif

	return 0;
}
