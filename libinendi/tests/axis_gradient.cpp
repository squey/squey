/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

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
	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(AXIS_INDEX));

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
	int last_green = 255;
	int last_red = 0;
	std::ifstream plotted_values(outputvalues);

	QColor qcolor;
	out.get_lines_properties().get_line_properties(0).toQColor(qcolor);
	PV_VALID(qcolor.green(), last_green);
	PV_VALID(qcolor.red(), last_red);
	out.get_lines_properties()
	    .get_line_properties(out.get_selection().count() - 1)
	    .toQColor(qcolor);
	PV_VALID(qcolor.green(), 16);
	PV_VALID(qcolor.red(), 255);

	for (size_t i = 0; i < out.get_selection().count(); i++) {

		PVCore::PVHSVColor color = out.get_lines_properties().get_line_properties(i);
		uint32_t wanted_color;

		plotted_values >> wanted_color;
		uint8_t actual_color = color.h();

		color.toQColor(qcolor);
		int green = qcolor.green();
		int red = qcolor.red();

		PV_ASSERT_VALID(green <= last_green);
		PV_ASSERT_VALID(red >= last_red);

		last_green = green;
		last_red = red;

		PV_VALID((size_t)actual_color, (size_t)wanted_color);
	}

#endif

	return 0;
}
