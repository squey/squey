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

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy.log";
static constexpr const char* csv_file2 = TEST_FOLDER "/sources/proxy_mineset.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
static constexpr const char* INVESTIGATION_PATH = "/tmp/tmp_investigation.pvi";
static constexpr unsigned int ROW_COUNT = 100000;
static constexpr unsigned int MINESET_ROW_COUNT = 1000;
#ifdef SQUEY_BENCH
static constexpr unsigned int dupl = 200;
#else
static constexpr unsigned int dupl = 1;
#endif

double save_investigation()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl);
	env.add_source(std::vector<std::string>{csv_file2, csv_file}, csv_file_format, dupl, false);

	size_t source_size = env.root.size<Squey::PVSource>();
	PV_VALID(source_size, 2UL);

	auto sources = env.root.get_children<Squey::PVSource>();
	auto it = sources.begin();
	PV_VALID((*it)->get_name(), std::string("proxy.log"));
	std::advance(it, 1);
	const std::string& source1 = std::string(TEST_FOLDER "/sources");
	const std::string& source2 = (*it)->get_name();
	PV_ASSERT_VALID(source1.compare(source1.length() - source2.length(), source2.length(), source2) == 0);

	env.compute_mappings();
	env.compute_plottings();
	env.compute_views();

	size_t mapped_size = env.root.size<Squey::PVMapped>();
	PV_VALID(mapped_size, 2UL);

	size_t plotted_size = env.root.size<Squey::PVPlotted>();
	PV_VALID(plotted_size, 2UL);

	size_t view_size = env.root.size<Squey::PVView>();
	PV_VALID(view_size, 2UL);
	auto view = env.root.get_children<Squey::PVView>().front();

	/**
	 * Add layers
	 */
	view->add_new_layer("layer #2");
	view->add_new_layer("layer #3");

	/**
	 * Color layer
	 */
	const PVRow row_count = view->get_row_count();
	Squey::PVSelection sel(row_count);
	sel.select_all();
	view->get_layer_stack().get_layer_n(2).get_lines_properties().selection_set_color(
	    sel, HSV_COLOR_GREEN);

	/**
	 *  Sort axes and remove last one
	 */
	// use of const_cast to get direct access to API (don't try this at home)
	auto& axes_comb =
	    const_cast<Squey::PVAxesCombination&>(view->get_axes_combination());
	axes_comb.sort_by_name();
	std::vector<PVCol> to_remove = {PVCol(14)};
	axes_comb.remove_axes(to_remove.begin(), to_remove.end());

	auto start = std::chrono::system_clock::now();

	PVCore::PVSerializeArchiveZip ar(INVESTIGATION_PATH, PVCore::PVSerializeArchive::write,
	                                 SQUEY_ARCHIVES_VERSION, false);
	env.root.save_to_file(ar);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	return diff.count();
}

double load_investigation()
{
	Squey::PVRoot root;

	auto start = std::chrono::system_clock::now();

	PVCore::PVSerializeArchiveZip ar(INVESTIGATION_PATH, PVCore::PVSerializeArchive::read,
	                                 SQUEY_ARCHIVES_VERSION, false);
	root.load_from_archive(ar);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	/**
	 * Check sources
	 */
	auto sources = root.get_children<Squey::PVSource>();
	PV_VALID(sources.size(), 2UL);

	auto it = sources.begin();
	PV_VALID((*it)->get_name(), std::string("proxy.log"));
	PV_VALID((*it)->get_row_count(), ROW_COUNT * dupl);
	std::advance(it, 1);
	PV_VALID((*it)->get_row_count(), ROW_COUNT * dupl + MINESET_ROW_COUNT);
	// FIXME : The name is the one from the archive. It should be a real name given by user.
	//	PV_VALID((*it)->get_name(), std::string(TEST_FOLDER "/sources"));

	/**
	 * Check mappeds
	 */
	size_t mapped_size = root.size<Squey::PVMapped>();
	PV_VALID(mapped_size, 2UL);

	/**
	 * Check plotteds
	 */
	size_t plotted_size = root.size<Squey::PVPlotted>();
	PV_VALID(plotted_size, 2UL);

	/**
	 * Check view
	 */
	auto views = root.get_children<Squey::PVView>();
	PV_VALID(views.size(), 2UL);
	auto view = views.front();
	PV_VALID(view->get_row_count(), ROW_COUNT * dupl);

	/**
	 * Check axes
	 */
	PV_VALID(view->get_axes_combination().get_nraw_names().size(), 15LL);
	auto axes = view->get_axes_names_list();
	PV_VALID(axes.size(), 14LL);
	constexpr const char* expected_axes_name[] = {
	    "domain",   "host",        "http_method", "http_status", "login_id", "mime_type",  "port",
	    "protocol", "result_code", "src_ip",      "subdomain",   "time",     "time_spent", "tld",
	    //"total_bytes" (removed)
	};
	for (int i = 0; i < axes.size(); i++) {
		PV_VALID(axes[i].toStdString(), std::string(expected_axes_name[i]));
	}

	/**
	 * Check layers
	 */
	PV_VALID(view->get_layer_stack().get_layer_count(), 3);
	PV_VALID(view->get_layer_stack().get_selected_layer_index(), 2);
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_name() == "layer #2");
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_selection().count() ==
	                ROW_COUNT * dupl);
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_name() == "layer #3");
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_selection().count() ==
	                ROW_COUNT * dupl);

	/**
	 * Check line properties
	 */
	PVCore::PVHSVColor const* colors =
	    view->get_layer_stack().get_layer_n(2).get_lines_properties().get_buffer();
	bool colors_ok = true;
	for (size_t i = 0; i < ROW_COUNT * dupl; i++) {
		colors_ok &= colors[i] == HSV_COLOR_GREEN;
	}
	PV_ASSERT_VALID(colors_ok);

	return diff.count();
}

int main()
{
	double saving_time = save_investigation();

#ifndef SQUEY_BENCH
	pvlogger::info() << "saving time took " << saving_time << " sec" << std::endl;
#endif

	double loading_time = load_investigation();

#ifndef SQUEY_BENCH
	pvlogger::info() << "loading time took " << loading_time << " sec" << std::endl;
#else
	std::cout << saving_time + loading_time;
#endif

	PVRush::PVNrawCacheManager::get().remove_nraws_from_investigation(INVESTIGATION_PATH);

	// Recheck loading without cache
	load_investigation();

	return 0;
}
