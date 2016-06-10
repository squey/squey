/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "common.h"

#include <inendi/PVLayerFilter.h>
#include <inendi/PVLayer.h>
#include <pvkernel/core/PVPercentRangeType.h>
#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVAxisIndexType.h>

#include <pvkernel/core/inendi_assert.h>

#include <chrono>
#include <fstream>
#include <string>

#ifndef INSPECTOR_BENCH
constexpr size_t CHECK_COUNT = 10000;
#endif

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

/**
 * Tested file have 4x 0, 3x < 12499, 2x 12499, 1x >12499.
 * We can deduce color and selection from this construction with selection on
 * 60 % to 80 %.
 */
void check_line_validity(Inendi::PVLayer const& out, size_t line)
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
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 92);
	} else if (line < 12500) {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 112);
	} else {
		PV_VALID((int)out.get_lines_properties().get_line_properties(line).h(), 59);
	}
}

int main()
{
// Init nraw
#ifdef INSPECTOR_BENCH
	std::string big_file_path;
	big_file_path.resize(L_tmpnam);
	// We assume that this name will not be use by another program before we
	// create it.
	tmpnam(&big_file_path.front());

	{
		std::ifstream ifs(filename);
		std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

		std::ofstream big_file(big_file_path);
		// Duplicate file to have one millions lines
		for (int i = 0; i < 20; i++) {
			big_file << content;
		}
	}
	pvtest::TestEnv env(big_file_path, fileformat);
	std::remove(big_file_path.c_str());
#else
	pvtest::TestEnv env(filename, fileformat);
#endif

	env.compute_mapping();
	env.compute_plotting();
	env.compute_views();
	Inendi::PVView* view = env.root->current_view();

	// Load every layer filter.
	Inendi::common::load_layer_filters();

	// Load heat line plugin
	Inendi::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name("frequency-gradient");
	Inendi::PVLayerFilter::p_type fclone = filter_org->clone<Inendi::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter("frequency-gradient");

	// Setup parameters.
	args["axes"].setValue(PVCore::PVAxisIndexType(1));
	auto scale = args["scale"].value<PVCore::PVEnumType>();
	scale.set_sel(1);
	args["scale"].setValue(scale);
	args["colors"].setValue(PVCore::PVPercentRangeType(0.6, 0.8));

	Inendi::PVLayer out("Out", view->get_row_count());
	Inendi::PVLayer& in = view->get_layer_stack_output_layer();

	fclone->set_view(view);
	fclone->set_output(&out);
	fclone->set_args(args);

	auto start = std::chrono::system_clock::now();

	// Run heat line computation
	fclone->operator()(in);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	// Check result
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();
	pvcop::db::array const& col = nraw.collection().column(1);

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
#endif

	return 0;
}
