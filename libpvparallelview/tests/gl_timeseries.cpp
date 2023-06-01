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

#include <squey/PVRangeSubSampler.h>

#include <squey/PVLayerFilter.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/core/squey_bench.h> // for BENCH_END, BENCH_START

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>

#include <pvcop/core/selected_array.h>
#include <pvcop/core/range.h>

#include <limits>
#include <random>

#include "common.h"
#include <omp.h>

#include <pvkernel/rush/PVNraw.h>
#include <QApplication>

#include <pvparallelview/PVSeriesView.h>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>

// datetime
#include <pvcop/types/datetime_us.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <pvcop/types/factory.h>

#include <pvlogger.h>

static constexpr const char* csv_file = TEST_FOLDER "/picviz/timeserie_fusion.csv";
static constexpr const char* csv_file_format = TEST_FOLDER "/picviz/timeserie_fusion.csv.format";

int main(int argc, char** argv)
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::Source);

	env.compute_mappings();
	env.compute_plottings();
	env.compute_views();

	auto plotteds = env.root.get_children<Squey::PVPlotted>();

	const auto& plotteds_vector = plotteds.front()->get_plotteds();

	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	view->select_all();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	QApplication a(argc, argv);

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (size_t i = 1; i < plotteds_vector.size(); i++) {
		timeseries.emplace_back(plotteds_vector[i].to_core_array<uint32_t>());
	}
	Squey::PVRangeSubSampler sampler(nraw.column(PVCol(0)), timeseries, nraw,
	                                  view->get_real_output_selection());

	std::unordered_set<size_t> selected_timeseries;
	for (size_t i = 0; i < timeseries.size(); ++i) {
		selected_timeseries.emplace(i);
	}
	sampler.set_selected_timeseries(selected_timeseries);

	sampler.set_sampling_count(1600);

	auto show_plot = [&](PVParallelView::PVSeriesView::Backend backend, bool exec = false) {
		PVParallelView::PVSeriesView plot(sampler, backend);

		plot.set_background_color(QColor(100, 10, 10, 255));

		std::vector<PVParallelView::PVSeriesView::SerieDrawInfo> series_draw_order;
		std::unordered_set<size_t> selected_timeseries;
		for (size_t i = 0; i < timeseries.size(); ++i) {
			series_draw_order.push_back(
			    {i, QColor(rand() % 156 + 100, rand() % 156 + 100, rand() % 156 + 100)});
		}
		plot.show_series(std::move(series_draw_order));

		plot.resize(1600, 900);

		sampler.resubsample();

		plot.grab();
		plot.refresh();
		plot.grab();
		sampler.resubsample();
		plot.grab();

		if (exec) {
			plot.show();
			a.exec();
		}
	};

	// show_plot(PVParallelView::PVSeriesView::Backend::QPainter);
	// show_plot(PVParallelView::PVSeriesView::Backend::OpenGL);
	show_plot(PVParallelView::PVSeriesView::Backend::OffscreenOpenGL, true);
}
