//
// MIT License
//
// © Squey, 2026
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

#include <pvparallelview/PVSeriesRendererHybrid.h>
#include <pvparallelview/PVSeriesRendererQPainter.h>
#include <pvparallelview/PVSeriesRendererRaster.h>
#ifdef SQUEY_SERIES_QRHI
#include <pvparallelview/PVSeriesRendererQRhi.h>
#endif
#include <squey/PVRangeSubSampler.h>
#include <squey/PVView.h>

#include <QGuiApplication>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "common.h"

/**
 * PVSeriesRendererRaster replaces the QPainter polylines by vertical spans it fills in
 * itself, so nothing but a picture comparison would catch it drifting away from the
 * reference renderer. The two do not agree pixel for pixel -- QPainter spreads a steep
 * segment over the two columns it joins, the rasteriser keeps it in the left one -- so
 * what is checked is that they light the same columns, at the same rows, to within the
 * one pixel that rounding can move.
 */

using namespace PVParallelView;
using PVRSS = Squey::PVRangeSubSampler;

static constexpr int image_width = 431; // deliberately not a multiple of the tile width
static constexpr int image_height = 173;

struct column_extent {
	int top = -1;
	int bottom = -1;
};

// Rows of the first and last lit pixel of every column, ignoring which serie lit them:
// with several series crossing each other that is all the two renderers can be held to.
static std::vector<column_extent> column_extents(QImage const& image, QRgb background)
{
	std::vector<column_extent> extents(image.width());
	for (int y = 0; y < image.height(); ++y) {
		for (int x = 0; x < image.width(); ++x) {
			if ((image.pixel(x, y) & 0xffffff) == (background & 0xffffff)) {
				continue;
			}
			if (extents[x].top < 0) {
				extents[x].top = y;
			}
			extents[x].bottom = y;
		}
	}
	return extents;
}

static void compare(QImage const& reference,
                    QImage const& raster,
                    PVSeriesView::DrawMode mode,
                    const char* what,
                    const char* backend)
{
	PV_ASSERT_VALID(not raster.isNull(), "mode", int(mode), "backend", std::string(backend));
	PV_ASSERT_VALID(reference.size() == raster.size(), "mode", int(mode), "kind",
	                std::string(what), "backend", std::string(backend));

	const auto ref_extents = column_extents(reference, qRgb(0, 0, 0));
	const auto ras_extents = column_extents(raster, qRgb(0, 0, 0));

	size_t lit_columns = 0;
	// The rightmost column is left out: in LinesAlways the reference closes its polyline
	// on a duplicate of its last point, and Qt drops the pixel of that zero-length
	// segment, so the last sample goes missing there. The rasteriser draws it.
	for (int x = 0; x < reference.width() - 1; ++x) {
		const bool ref_lit = ref_extents[x].top >= 0;
		const bool ras_lit = ras_extents[x].top >= 0;
		PV_ASSERT_VALID(ref_lit == ras_lit, "mode", int(mode), "kind", std::string(what), "column",
		                x, "backend", std::string(backend));
		if (not ref_lit) {
			continue;
		}
		++lit_columns;
		PV_ASSERT_VALID(std::abs(ref_extents[x].top - ras_extents[x].top) <= 1, "mode", int(mode),
		                "kind", std::string(what), "column", x, "reference top",
		                ref_extents[x].top, "other top", ras_extents[x].top, "backend",
		                std::string(backend));
		PV_ASSERT_VALID(std::abs(ref_extents[x].bottom - ras_extents[x].bottom) <= 1, "mode",
		                int(mode), "kind", std::string(what), "column", x, "reference bottom",
		                ref_extents[x].bottom, "other bottom", ras_extents[x].bottom, "backend",
		                std::string(backend));
	}

	// A renderer that draws nothing would sail through the loop above. The narrow range
	// only holds a few dozen samples, hence the low bar.
	PV_ASSERT_VALID(lit_columns >= 10, "mode", int(mode), "kind", std::string(what),
	                "lit columns", lit_columns, "backend", std::string(backend));
}

int main(int argc, char** argv)
{
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

	// After TestEnv, which runs a QCoreApplication of its own for the length of its
	// constructor: the GPU backend needs a GUI application to reach a driver at all, and
	// two applications may not overlap.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	QGuiApplication app(argc, argv);

	PVRSS sampler(nraw.column(PVCol(1)), timeseries, nraw, view->get_real_output_selection(),
	              nullptr, image_width);
	view->select_all();

	std::vector<PVSeriesView::SerieDrawInfo> draw_order;
	std::unordered_set<size_t> selected;
	for (size_t i = 2; i < std::min<size_t>(timeseries.size(), 10); ++i) {
		selected.insert(i);
		draw_order.push_back({i, QColor::fromHsv(int((i * 61) % 360), 200, 255)});
	}
	sampler.set_selected_timeseries(selected);

	PVSeriesRendererQPainter reference(sampler);
	PVSeriesRendererRaster raster(sampler);
	std::vector<std::pair<PVSeriesAbstractRenderer*, const char*>> renderers = {
	    {&reference, "qpainter"}, {&raster, "raster"}};

#ifdef SQUEY_SERIES_QRHI
	// The GPU backend is only exercised where a device actually answers: a build machine
	// or a headless runner has none, and its absence is not a test failure.
	PVSeriesRendererQRhi qrhi(sampler);
	const bool qrhi_available = PVSeriesRendererQRhi::capability();
	if (qrhi_available) {
		renderers.emplace_back(&qrhi, "qrhi");
	}
	printf("QRhi backend available: %s\n", qrhi_available ? "yes" : "no");
#endif

	for (auto const& [renderer, name] : renderers) {
		(void)name;
		renderer->resize(QSize(image_width, image_height));
		renderer->set_background_color(Qt::black);
		renderer->show_series(draw_order);
	}

	// Two zoom levels : the whole range, where every sample holds a value, and a narrow
	// one, where ranges fall empty and the gap handling of each mode comes into play.
	struct {
		double first;
		double last;
		const char* what;
	} const zooms[] = {{0., 1., "full range"}, {0.2, 0.22, "narrow range"}};

	for (auto const& zoom : zooms) {
		sampler.subsample(zoom.first, zoom.last);

		// The sampler must have bucketed the rows it was given. Checked here rather than
		// left to the picture: an empty histogram draws exactly like a renderer that has
		// stopped working, and telling the two apart from a blank plot cost a round trip
		// through a Windows runner once already.
		const auto& hist = sampler.histogram();
		PV_ASSERT_VALID(std::accumulate(hist.begin(), hist.end(), size_t(0)) > 0, "kind",
		                std::string(zoom.what));
		for (PVSeriesView::DrawMode mode :
		     {PVSeriesView::DrawMode::Lines, PVSeriesView::DrawMode::Points,
		      PVSeriesView::DrawMode::LinesAlways}) {
			for (auto const& [renderer, name] : renderers) {
				renderer->set_draw_mode(mode);
			}
			// The reference is only held to the shape of what it draws. Every backend
			// past the rasteriser has to match the rasteriser itself pixel for pixel:
			// which one is in use must not be something the eye can tell.
			const QImage expected = reference.grab();
			const QImage rasterised = raster.grab();
			compare(expected, rasterised, mode, zoom.what, "raster");
			for (size_t i = 2; i < renderers.size(); ++i) {
				const QImage other = renderers[i].first->grab();
				PV_ASSERT_VALID(other.convertToFormat(QImage::Format_RGB32) ==
				                    rasterised.convertToFormat(QImage::Format_RGB32),
				                "mode", int(mode), "kind", std::string(zoom.what), "backend",
				                std::string(renderers[i].second));
			}
		}
	}

	// PVSeriesRendererHybrid only reaches for the GPU past a total amount of work
	// (samples shown times series drawn) -- replay the full range with enough repeated
	// series, at the width used everywhere else in this file, to comfortably clear that
	// threshold, and check the picture still matches the rasteriser exactly. Whichever
	// renderer Hybrid picked, on this machine, must not be visible in the output.
	{
		std::vector<PVSeriesView::SerieDrawInfo> heavy_draw_order;
		auto it = selected.begin();
		const size_t heavy_series_count =
		    (PVSeriesRendererHybrid::high_threshold + image_width - 1) / image_width + 8;
		for (size_t i = 0; i < heavy_series_count; ++i) {
			if (it == selected.end()) {
				it = selected.begin();
			}
			heavy_draw_order.push_back({*it++, QColor::fromHsv(int((i * 37) % 360), 200, 255)});
		}

		PVSeriesRendererRaster heavy_raster(sampler);
		PVSeriesRendererHybrid hybrid(sampler);
		for (PVSeriesAbstractRenderer* renderer :
		     {(PVSeriesAbstractRenderer*)&heavy_raster, (PVSeriesAbstractRenderer*)&hybrid}) {
			renderer->resize(QSize(image_width, image_height));
			renderer->set_background_color(Qt::black);
			renderer->set_draw_mode(PVSeriesView::DrawMode::Lines);
			renderer->show_series(heavy_draw_order);
		}
		sampler.subsample(0., 1.);

		const QImage expected = heavy_raster.grab();
		const QImage got = hybrid.grab();
		PV_ASSERT_VALID(got.convertToFormat(QImage::Format_RGB32) ==
		                    expected.convertToFormat(QImage::Format_RGB32),
		                "heavy series count", heavy_series_count);
#ifdef SQUEY_SERIES_QRHI
		// Where a device actually answers, this scale must be the one place in this file
		// where the two pictures agree only because the GPU path was checked, not because
		// Hybrid quietly stayed on the rasteriser the whole time.
		if (PVSeriesRendererQRhi::capability()) {
			PV_ASSERT_VALID(hybrid.used_gpu_last_frame(), "heavy series count",
			                heavy_series_count);
		}
#endif
	}

	return 0;
}
