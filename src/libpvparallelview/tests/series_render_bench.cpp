//
// Throwaway benchmark harness for the series renderers (not part of the test suite).
//
#include <pvparallelview/PVSeriesRendererQPainter.h>
#include <pvparallelview/PVSeriesRendererRaster.h>
#ifdef SQUEY_SERIES_QRHI
#include <pvparallelview/PVSeriesRendererQRhi.h>
#endif

#include <QGuiApplication>
#include <squey/PVRangeSubSampler.h>
#include <squey/PVView.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "common.h"

using namespace PVParallelView;
using clk = std::chrono::steady_clock;

static double best_of(PVSeriesAbstractRenderer& r, int reps)
{
	double best = 1e18;
	for (int i = 0; i < reps; ++i) {
		auto t = clk::now();
		volatile auto image = r.grab();
		(void)image;
		best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t).count());
	}
	return best;
}

int main(int argc, char** argv)
{
	const int w = argc > 1 ? atoi(argv[1]) : 1600;
	const int h = argc > 2 ? atoi(argv[2]) : 900;

	pvtest::TestEnv env(TEST_FOLDER "/picviz/timeserie_fusion.csv",
	                    TEST_FOLDER "/picviz/timeserie_fusion.csv.format", 1,
	                    pvtest::ProcessUntil::View);
	auto scaleds = env.root.get_children<Squey::PVScaled>();
	const auto& scaleds_vector = scaleds.front()->get_scaleds();
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (const auto& scaled : scaleds_vector) {
		timeseries.emplace_back(scaled.to_core_array<uint32_t>());
	}

	Squey::PVRangeSubSampler sampler(nraw.column(PVCol(1)), timeseries, nraw,
	                                 view->get_real_output_selection(), nullptr, size_t(w));
	view->select_all();
	std::unordered_set<size_t> selected;
	for (size_t i = 2; i < timeseries.size(); ++i) {
		selected.insert(i);
	}
	sampler.set_selected_timeseries(selected);
	sampler.subsample(0., 1.);

	QGuiApplication app(argc, argv);
#ifdef SQUEY_SERIES_QRHI
	const bool qrhi = PVSeriesRendererQRhi::capability();
#else
	const bool qrhi = false;
#endif

	printf("%dx%d, %zu distinct series repeated, QRhi=%s\n", w, h, selected.size(),
	       qrhi ? "yes" : "no");
	printf("%8s %14s %14s %10s %12s\n", "drawn", "QPainter(ms)", "raster(ms)", "speedup",
	       "qrhi(ms)");
	for (int drawn : {1, 4, 16, 64, 256, 1024}) {
		std::vector<PVSeriesView::SerieDrawInfo> draw_order;
		auto it = selected.begin();
		for (int i = 0; i < drawn; ++i) {
			if (it == selected.end()) {
				it = selected.begin();
			}
			draw_order.push_back({*it++, QColor::fromHsv((i * 37) % 360, 200, 255)});
		}

		PVSeriesRendererQPainter reference(sampler);
		PVSeriesRendererRaster raster(sampler);
		for (PVSeriesAbstractRenderer* r :
		     {(PVSeriesAbstractRenderer*)&reference, (PVSeriesAbstractRenderer*)&raster}) {
			r->resize(QSize(w, h));
			r->set_background_color(Qt::black);
			r->set_draw_mode(PVSeriesView::DrawMode::Lines);
			r->show_series(draw_order);
		}
		const double q = best_of(reference, 5);
		const double s = best_of(raster, 5);
		double g = -1.0;
#ifdef SQUEY_SERIES_QRHI
		if (qrhi) {
			PVSeriesRendererQRhi gpu(sampler);
			gpu.resize(QSize(w, h));
			gpu.set_background_color(Qt::black);
			gpu.set_draw_mode(PVSeriesView::DrawMode::Lines);
			gpu.show_series(draw_order);
			g = best_of(gpu, 5);
		}
#endif
		printf("%8d %14.2f %14.2f %9.1fx %12.2f\n", drawn, q, s, q / s, g);
		fflush(stdout);
	}
	return 0;
}
