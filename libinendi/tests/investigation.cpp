/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_assert.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
static constexpr const char* INVESTIGATION_PATH = "/tmp/tmp_investigation.pvi";
static constexpr unsigned int ROW_COUNT = 100000;
#ifdef INSPECTOR_BENCH
    static constexpr unsigned int dupl = 200;
#else
    static constexpr unsigned int dupl = 1;
#endif


double save_investigation()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl);

	auto sources = env.root->get_children<Inendi::PVSource>();
	PV_VALID(sources.size(), 1);
	auto source = sources[0];

	env.compute_mapping();
	env.compute_plotting();

	auto views = source->get_children<Inendi::PVView>();
	PV_VALID(views.size(), 1);
	auto view = views[0];

	/**
	 * Add layers
	 */
	view->add_new_layer("layer #2");
	view->add_new_layer("layer #3");

	/**
	 * Color layer
	 */
	const PVRow row_count = view->get_row_count();
	Inendi::PVSelection sel(row_count);
	sel.select_all();
	view->get_layer_stack().get_layer_n(2).get_lines_properties().selection_set_color(sel, row_count, HSV_COLOR_GREEN);

	/**
	 *  Sort axes and remove last one
	 */
	// use of const_cast to get direct access to API (don't try this at home)
	Inendi::PVAxesCombination& axes_comb = const_cast<Inendi::PVAxesCombination&>(view->get_axes_combination());
	axes_comb.sort_by_name();
	axes_comb.remove_axis(14);

	auto start = std::chrono::system_clock::now();

	env.root->save_to_file(INVESTIGATION_PATH);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	return diff.count();
}

double load_investigation()
{
	Inendi::PVRoot_p root(new Inendi::PVRoot());

	auto start = std::chrono::system_clock::now();

	root->load_from_file(INVESTIGATION_PATH);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	/**
	 * Check source
	 */
	auto sources = root->get_children<Inendi::PVSource>();
	PV_VALID(sources.size(), 1);
	auto source = sources[0];
	source->load_from_disk();
	source->process_from_source();

	const PVRow row_count = source->get_row_count();
	PV_VALID(row_count, ROW_COUNT * dupl);

	/**
	 * Check view
	 */
	auto views = source->get_children<Inendi::PVView>();
	PV_VALID(views.size(), 1);
	auto view = views[0];
	PV_VALID(view->get_row_count(), ROW_COUNT * dupl);

	/**
	 * Check axes
	 */
	PV_VALID(view->get_original_axes_names_list().size(), 15);
	auto axes = view->get_axes_names_list();
	PV_VALID(axes.size(), 14);
	constexpr const char* expected_axes_name[] = {
		"domain",
		"host",
		"http_method",
		"http_status",
		"login_id",
		"mime_type",
		"port",
		"protocol",
		"result_code",
		"src_ip",
		"subdomain",
		"time",
		"time_spent",
		"tld",
		//"total_bytes" (removed)
	};
	for (int i = 0; i < axes.size(); i++) {
		PV_ASSERT_VALID(axes[i] == QString(expected_axes_name[i]));
	}

	/**
	 * Check layers
	 */
	PV_VALID(view->get_layer_stack().get_layer_count(), 3);
	PV_VALID(view->get_layer_stack().get_selected_layer_index(), 2);
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_name() == "layer #2");
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_selection().count() == row_count);
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_name() == "layer #3");
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_selection().count() == row_count);

	/**
	 * Check line properties
	 */
	PVCore::PVHSVColor const* colors = view->get_layer_stack().get_layer_n(2).get_lines_properties().get_buffer();
	bool colors_ok = true;
	for (size_t i = 0; i < row_count; i++) {
		colors_ok &= colors[i] == HSV_COLOR_GREEN;
	}
	PV_ASSERT_VALID(colors_ok);

	return diff.count();
}

int main()
{
	double saving_time = save_investigation();

#ifndef INSPECTOR_BENCH
		pvlogger::info() << "saving time took " << saving_time << " sec" << std::endl;
#endif

	double loading_time = load_investigation();

#ifndef INSPECTOR_BENCH
		pvlogger::info() << "loading time took " << loading_time << " sec" << std::endl;
#else
		std::cout << saving_time + loading_time;
#endif

    return 0;
}
