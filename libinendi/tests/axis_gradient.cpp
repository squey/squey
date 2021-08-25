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

#include <inendi/PVLayerFilter.h>
#include <inendi/PVLayer.h>

#include <pvkernel/core/PVOriginalAxisIndexType.h>

#include <pvkernel/core/inendi_assert.h>

#include <chrono>
#include <fstream>
#include <string>

#ifndef INSPECTOR_BENCH
constexpr size_t ndup = 1;
#else
constexpr size_t ndup = 20;
#endif

const std::string filename = TEST_FOLDER "/picviz/axis_gradient.csv";
const std::string fileformat = TEST_FOLDER "/picviz/axis_gradient.csv.format";
const std::string outputvalues = TEST_FOLDER "/picviz/axis_gradient.colors";

static constexpr const char* PLUGIN_NAME = "axis-gradient";
static constexpr const int AXIS_INDEX = 0;

int main()
{
	// Init nraw
	pvtest::TestEnv env(filename, fileformat, ndup, pvtest::ProcessUntil::View);

	Inendi::PVView* view = env.root.current_view();

	// Load every layer filter.
	Inendi::common::load_layer_filters();

	// Load axis gradient line plugin
	Inendi::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(PLUGIN_NAME);
	Inendi::PVLayerFilter::p_type fclone = filter_org->clone<Inendi::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter(PLUGIN_NAME);

	// Setup parameters.
	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(PVCol(AXIS_INDEX)));

	Inendi::PVLayer out("Out", view->get_row_count());
	Inendi::PVLayer& in = view->get_layer_stack_output_layer();

	fclone->set_view(view);
	fclone->set_output(&out);
	fclone->set_args(args);

	auto start = std::chrono::system_clock::now();

	// Run plugin computation
	fclone->operator()(in);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Check result against reference file
	int last_blue = 255;
	int last_red = 0;
	std::ifstream plotted_values(outputvalues);

	QColor qcolor = out.get_lines_properties().get_line_properties(0).toQColor();
	PV_VALID(qcolor.blue(), last_blue);
	PV_VALID(qcolor.red(), last_red);
	qcolor =
	    out.get_lines_properties().get_line_properties(out.get_selection().count() - 1).toQColor();
	PV_VALID(qcolor.blue(), 0);
	PV_VALID(qcolor.red(), 255);

	for (size_t i = 0; i < out.get_selection().count(); i++) {

		PVCore::PVHSVColor color = out.get_lines_properties().get_line_properties(i);
		uint32_t wanted_color;

		plotted_values >> wanted_color;
		uint8_t actual_color = color.h();

		qcolor = color.toQColor();
		int blue = qcolor.blue();
		int red = qcolor.red();

		PV_ASSERT_VALID(blue <= last_blue);
		PV_ASSERT_VALID(red >= last_red);

		last_blue = blue;
		last_red = red;

		PV_VALID((size_t)actual_color, (size_t)wanted_color);
	}

#endif

	return 0;
}
