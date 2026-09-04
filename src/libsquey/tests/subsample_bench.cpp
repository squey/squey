//
// Throwaway benchmark harness for PVRangeSubSampler (not part of the test suite).
//
#include <squey/PVRangeSubSampler.h>

#include "common.h"

#include <chrono>
#include <cstdio>
#include <numeric>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv)
{
	const char* file = argc > 1 ? argv[1] : "/tmp/seriesbench/data/ts_sorted.csv";
	const size_t sampling_count = argc > 2 ? std::stoul(argv[2]) : 1600;
	const size_t nseries = argc > 3 ? std::stoul(argv[3]) : 4;

	pvtest::TestEnv env(file, std::string(file) + ".format", 1, pvtest::ProcessUntil::View);
	auto scaleds = env.root.get_children<Squey::PVScaled>();
	const auto& scaleds_vector = scaleds.front()->get_scaleds();
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (const auto& i : scaleds_vector) {
		timeseries.emplace_back(i.to_core_array<uint32_t>());
	}

	auto t0 = clk::now();
	Squey::PVRangeSubSampler sampler(nraw.column(PVCol(0)), timeseries, nraw,
	                                 view->get_real_output_selection());
	view->select_all();
	double ctor_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

	std::unordered_set<size_t> selected;
	for (size_t i = 1; i <= std::min(nseries, timeseries.size() - 1); ++i) {
		selected.insert(i);
	}
	sampler.set_sampling_count(sampling_count);
	sampler.set_selected_timeseries(selected);
	sampler.resubsample();

	// steady state : what a zoom or a pan costs
	double best = 1e18;
	for (int r = 0; r < 5; ++r) {
		auto t = clk::now();
		sampler.subsample(0.1 + 0.001 * r, 0.9 - 0.001 * r);
		best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t).count());
	}
	printf("%-42s rows=%zu series=%zu samples=%zu | ctor+sort=%8.1f ms | subsample=%8.1f ms\n",
	       file, size_t(nraw.row_count()), selected.size(), sampling_count, ctor_ms, best);

	// what a window resize costs (set_sampling_count invalidates everything)
	best = 1e18;
	for (int r = 0; r < 5; ++r) {
		auto t = clk::now();
		sampler.set_sampling_count(sampling_count + r + 1);
		sampler.resubsample();
		best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t).count());
	}
	printf("%-42s %54s resize=%8.1f ms\n", "", "", best);
	return 0;
}
