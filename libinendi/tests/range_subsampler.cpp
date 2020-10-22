/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */
#include <inendi/PVRangeSubSampler.h>

#include "common.h"
#include <pvkernel/core/inendi_assert.h>

#include <algorithm>

#include <pvlogger.h>

#include <fstream>

using display_type = typename Inendi::PVRangeSubSampler::display_type;

struct testcase {
	std::string file_path;
	std::string format_path;
	PVCol col_time;
	PVCol col_ts;
	std::vector<size_t> sampling_counts;
	std::vector<std::function<void(Inendi::PVSelection&)>> apply_sel;
};

void test(const testcase& test)
{
	pvtest::TestEnv env(test.file_path, test.format_path, 1, pvtest::ProcessUntil::View);

	auto plotteds = env.root.get_children<Inendi::PVPlotted>();

	const auto& plotteds_vector = plotteds.front()->get_plotteds();

	Inendi::PVView* view = env.root.get_children<Inendi::PVView>().front();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (size_t i = 0; i < plotteds_vector.size(); i++) {
		timeseries.emplace_back(plotteds_vector[i].to_core_array<uint32_t>());
	}

	Inendi::PVRangeSubSampler sampler(nraw.column(test.col_time), timeseries, nraw,
	                                  view->get_real_output_selection());

	view->select_all();

	PV_VALID(sampler.valid(), false);

	for (size_t i = 0; i < test.sampling_counts.size(); i++) {
		test.apply_sel[i](const_cast<Inendi::PVSelection&>(view->get_real_output_selection()));

		const auto& sampling_count = test.sampling_counts[i];
		sampler.set_sampling_count(sampling_count);

		// select all timeseries
		std::unordered_set<size_t> selected_ts;
		std::generate_n(std::inserter(selected_ts, selected_ts.end()), timeseries.size(),
		                [x = 0]() mutable { return x++; });
		sampler.set_selected_timeseries(selected_ts);

		sampler.resubsample();
		const auto& avg_ts = sampler.sampled_timeserie(test.col_ts);

		PV_VALID(sampler.valid(), true);
		PV_VALID(sampler.samples_count(), sampling_count);
		PV_VALID(avg_ts.size(), sampling_count);

		PVRow indexes_count =
		    std::accumulate(sampler.histogram().begin(), sampler.histogram().end(), 0);
		PV_VALID(indexes_count, nraw.row_count());

		std::ifstream f;
		f.open(test.file_path + ".output_ranges_" + std::to_string(sampling_count));
		for (const auto& v : sampler.histogram()) {
			size_t range_values_count;
			f >> range_values_count;
			PV_VALID(range_values_count, v);
		}
		f.close();

		f.open(test.file_path + ".output_averages_" + std::to_string(sampling_count));
		for (const auto& v : avg_ts) {
			display_type avg_values_count;
			f >> avg_values_count;
			PV_VALID(avg_values_count, v);
		}
		f.close();
	}
}

int main()
{
	std::vector<testcase> testsuite = {
	    {TEST_FOLDER "/picviz/timeserie_iota.csv",
	     TEST_FOLDER "/picviz/timeserie_iota.csv.format",
	     PVCol(0),
	     PVCol(1),
	     {100, 200}, /* sampling_counts */
	     {[](auto&) {}, [](auto& s) { s.select_odd(); }}},

	    {TEST_FOLDER "/picviz/timeserie_fusion.csv",
	     TEST_FOLDER "/picviz/timeserie_fusion.csv.format",
	     PVCol(1),
	     PVCol(19),
	     {666, 333}, /* sampling_counts */
	     {[](auto& s) { s.select_even(); }, [](auto&) {}}},

	    {TEST_FOLDER "/picviz/timeserie_westmill.csv",
	     TEST_FOLDER "/picviz/timeserie_westmill.csv.format",
	     PVCol(1),
	     PVCol(8),
	     {773, 89}, /* sampling_counts */
	     {[](auto& s) { s.select_none(); }, [](auto& s) { s.select_odd(); }}}};

	for (const testcase& t : testsuite) {
		test(t);
	}
}
