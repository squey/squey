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

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file2 = TEST_FOLDER "/sources/proxy_mineset.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
static constexpr const char* INVESTIGATION_PATH = "/tmp/tmp_investigation.pvi";
static constexpr const char* ref_mapped_file = TEST_FOLDER "/picviz/ref_mapped";
static constexpr const char* ref_plotted_file = TEST_FOLDER "/picviz/ref_plotted";
static constexpr unsigned int ROW_COUNT = 100000;
#ifdef INSPECTOR_BENCH
static constexpr unsigned int dupl = 200;
#else
static constexpr unsigned int dupl = 1;
#endif

double save_investigation()
{
	// Check multiple sources in multipls scene
	pvtest::TestEnv env(csv_file, csv_file_format, dupl);
	env.add_source(csv_file, csv_file_format, dupl, true);
	env.add_source(csv_file2, csv_file_format, dupl, false);

	size_t source_size = env.root.size<Inendi::PVSource>();
	PV_VALID(source_size, 3UL);

	// Check scene name saving
	auto scenes = env.root.get_children<Inendi::PVScene>();
	auto it2 = scenes.begin();
	PV_VALID((*it2)->get_name(), std::string("scene"));
	std::advance(it2, 1);
	(*it2)->set_name("my  super name");
	PV_VALID((*it2)->get_name(), std::string("my  super name"));

	auto sources = env.root.get_children<Inendi::PVSource>();
	auto it = sources.begin();
	PV_VALID((*it)->get_name(), std::string("proxy_1bad.log"));
	std::advance(it, 1);
	PV_VALID((*it)->get_name(), std::string("proxy_mineset.log"));
	std::advance(it, 1);
	PV_VALID((*it)->get_name(), std::string("proxy_1bad.log"));

	env.compute_mappings();
	env.compute_plottings();
	env.compute_views();

	auto mappeds = env.root.get_children<Inendi::PVMapped>();
	PV_VALID(mappeds.size(), 3UL);
	auto* mapped = mappeds.front();
	mapped->set_name("other");
	PV_VALID(mapped->get_name(), std::string("other"));

	auto plotteds = env.root.get_children<Inendi::PVPlotted>();
	PV_VALID(plotteds.size(), 3UL);
	auto* plotted = plotteds.front();
	plotted->set_name("my plotting name");
	PV_VALID(plotted->get_name(), std::string("my plotting name"));

	size_t view_size = env.root.size<Inendi::PVView>();
	PV_VALID(view_size, 3UL);
	auto view = env.root.get_children<Inendi::PVView>().front();

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
	view->get_layer_stack().get_layer_n(2).get_lines_properties().selection_set_color(
	    sel, HSV_COLOR_GREEN);

	/**
	 *  Sort axes and remove last one
	 */
	// use of const_cast to get direct access to API (don't try this at home)
	auto& axes_comb =
	    const_cast<Inendi::PVAxesCombination&>(view->get_axes_combination());
	axes_comb.sort_by_name();
	std::vector<PVCol> to_remove = {PVCol(14)};
	axes_comb.remove_axes(to_remove.begin(), to_remove.end());

	auto start = std::chrono::system_clock::now();

	PVCore::PVSerializeArchiveZip ar(INVESTIGATION_PATH, PVCore::PVSerializeArchive::write,
	                                 INENDI_ARCHIVES_VERSION, true);
	env.root.save_to_file(ar);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	return diff.count();
}

double load_investigation()
{
	Inendi::PVRoot root;

	auto start = std::chrono::system_clock::now();

	{
		PVCore::PVSerializeArchiveZip ar(INVESTIGATION_PATH, PVCore::PVSerializeArchive::read,
		                                 INENDI_ARCHIVES_VERSION, true);
		root.load_from_archive(ar);
	}

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	/**
	 * Check scenes
	 */
	auto scenes = root.get_children<Inendi::PVScene>();
	auto it2 = scenes.begin();
	PV_VALID((*it2)->get_name(), std::string("scene"));
	std::advance(it2, 1);
	PV_VALID((*it2)->get_name(), std::string("my  super name"));

	/**
	 * Check sources
	 */
	auto sources = root.get_children<Inendi::PVSource>();
	PV_VALID(sources.size(), 3UL);
	auto source = sources.front();
	PV_VALID(source->get_format().is_valid(), true);
	PV_VALID(source->get_invalid_evts().size(), 0UL);
	// PV_VALID(source->get_invalid_evts().begin()->first, 0UL);
	// PV_VALID(source->get_invalid_evts().begin()->second,
	//          std::string("343,10.107.73.75,TCP_CLIENT_REFRESH_MISS,200,4420,GET,http,updates,"
	//                      "updates.copernic.com,copernic.com,com,80,5986,application/octet-stream"));

	auto it = sources.begin();
	PV_VALID((*it)->get_name(), std::string("proxy_1bad.log"));
	std::advance(it, 1);
	PV_VALID((*it)->get_name(), std::string("proxy_mineset.log"));
	std::advance(it, 1);
	PV_VALID((*it)->get_name(), std::string("proxy_1bad.log"));

	const PVRow row_count = source->get_row_count();
	PV_VALID(row_count, ROW_COUNT * dupl);

	/**
	 * Check Nraw
	 */
	PVRush::PVNraw const& nraw = source->get_rushnraw();
	PV_VALID(nraw.valid_rows_sel().bit_count(), size_t(ROW_COUNT));
	PV_VALID(nraw.valid_rows_sel().count(), ROW_COUNT);
	PV_VALID(nraw.valid_rows_sel().get_line(0), true);
	PV_VALID(nraw.get_valid_row_count(), size_t(ROW_COUNT));

	PV_VALID(nraw.column(PVCol(2)).at(24), std::string("toto"));
	PV_VALID(nraw.column(PVCol(2)).at(12), std::string());
	PV_VALID(nraw.column(PVCol(2)).at(13), std::string());

	/**
	 * Check mappeds
	 */
	auto mappeds = root.get_children<Inendi::PVMapped>();
	PV_VALID(mappeds.size(), 3UL);
	auto* mapped = mappeds.front();

	PV_VALID(mapped->get_name(), std::string("other"));

	pvcop::db::array const& mapping_values = mapped->get_column(PVCol(2));
	auto mapping = mapping_values.to_core_array<uint32_t>();
	std::ifstream ref_stream(ref_mapped_file);
	size_t i = 0;
	for (uint32_t v : mapping) {
		uint32_t ref;
		ref_stream >> ref;
		if (i++ > 0) {
			PV_VALID(ref, v);
		}
	}

	PV_VALID(mapped->get_properties_for_col(PVCol(0)).get_mode(), std::string("default"));
	auto const& minmax = mapped->get_properties_for_col(PVCol(0)).get_minmax();
	auto core_minmax = minmax.to_core_array<string_index_t>();
	PV_VALID(core_minmax[0], (string_index_t)0U);
	PV_VALID(core_minmax[1], (string_index_t)90038U);

	/**
	 * Check plotteds
	 */
	auto plotteds = root.get_children<Inendi::PVPlotted>();
	PV_VALID(plotteds.size(), 3UL);
	auto const* plotted = plotteds.front();
	PV_VALID(plotted->get_name(), std::string("my plotting name"));

	uint32_t const* plotting_values = plotted->get_column_pointer(PVCol(0));
	std::ifstream ref_plotted_stream(ref_plotted_file);
	for (size_t i = 0; i < plotted->get_row_count(); i++) {
		uint32_t ref;
		ref_plotted_stream >> ref;
		if (ref != plotting_values[i]) {
			pvlogger::info() << "#" << i << std::endl;
		}
		if (i > 0) {
			PV_VALID(ref, plotting_values[i]);
		}
	}

	PV_VALID(plotted->get_properties_for_col(PVCol(2)).get_mode(), std::string("default"));
	PV_VALID(plotted->get_col_min_row(PVCol(2)), 75173U);
	PV_VALID(plotted->get_col_max_row(PVCol(2)), 0U);

	/**
	 * Check view
	 */
	auto views = root.get_children<Inendi::PVView>();
	PV_VALID(views.size(), 3UL);
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
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_selection().count() == row_count);
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_name() == "layer #3");
	PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(2).get_selection().count() == row_count);

	/**
	 * Check line properties
	 */
	PVCore::PVHSVColor const* colors =
	    view->get_layer_stack().get_layer_n(2).get_lines_properties().get_buffer();
	for (size_t i = 0; i < row_count; i++) {
		PV_VALID(colors[i].h(), (uint8_t)HSV_COLOR_GREEN.h());
		PV_VALID(view->get_color_in_output_layer(i).h(), (uint8_t)HSV_COLOR_GREEN.h());
	}

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

	PVRush::PVNrawCacheManager::get().remove_nraws_from_investigation(INVESTIGATION_PATH);

	// Recheck loading without cache
	load_investigation();

	return 0;
}
