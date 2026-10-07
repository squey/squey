/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Lines drawn by density.
//
// A line of the parallel view stands for the rows of its bucket. Drawn by density,
// it is as opaque as those rows laid over one another: a bundle of many rows stands
// out, a row on its own fades. Turned on from the toolbar, the opacity of a single
// row's line starts at one suiting the data; turned off, the lines are opaque
// again, and the rows are no longer counted.
//
// The view outlives the rendering context here, as it does when the whole
// subsystem is torn down first: the scene lets go of the counting as the context
// goes, rather than when it is itself destroyed, once the context is gone.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelViewParamsWidget.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVViewRenderingContext.h>
#include <pvparallelview/PVZonesManager.h>

#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <QAction>
#include <QApplication>
#include <QElapsedTimer>
#include <QImage>
#include <QTemporaryDir>
#include <QThread>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "common.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="left" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="right" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

//! Rows drawing one same line.
constexpr PVRow BUNDLE = 900;
//! Rows drawing a line each.
constexpr PVRow SCATTERED = 1000;

/**
 * Run the event loop for a while.
 *
 * Sleeping in between, not only pumping: the views draw on threads of their own and
 * post the result back, and pumping alone returns as soon as nothing is queued.
 */
void pump(int ms)
{
	QElapsedTimer waited;
	waited.start();
	while (waited.elapsed() < ms) {
		QApplication::processEvents(QEventLoop::AllEvents, 5);
		QThread::msleep(2);
	}
}

//! How many pixels of the selected lines the only zone has, by opacity.
std::array<size_t, 256> opacities(PVParallelView::PVFullParallelScene& scene)
{
	PVParallelView::PVLinesView& lines = scene.get_lines_view();
	const QImage image =
	    lines.get_single_zone_images(lines.get_zone_index_offset(0)).sel->qimage().copy();

	std::array<size_t, 256> pixels{};
	for (int y = 0; y < image.height(); ++y) {
		const auto* line = reinterpret_cast<const QRgb*>(image.constScanLine(y));
		for (int x = 0; x < image.width(); ++x) {
			++pixels[qAlpha(line[x])];
		}
	}
	return pixels;
}

//! The pixels drawn at all, however faintly.
size_t drawn(std::array<size_t, 256> const& pixels)
{
	return std::accumulate(pixels.begin() + 1, pixels.end(), size_t(0));
}

QAction* toolbar_action(PVParallelView::PVFullParallelView& widget, QString const& text)
{
	auto* params = widget.findChild<PVParallelView::PVFullParallelViewParamsWidget*>();
	PV_ASSERT_VALID(params != nullptr, "the toolbar", "is not in the view");

	for (QAction* action : params->actions()) {
		if (action->text() == text) {
			return action;
		}
	}
	PV_ASSERT_VALID(false, "the toolbar has no action", text.toStdString());
	return nullptr;
}

std::vector<uint32_t> const& selected_rows_per_line(PVParallelView::PVViewRenderingContext& context)
{
	PVParallelView::PVZonesManager& zm = context.get_zones_manager();
	return zm.get_zone_tree(zm.get_zone_id(0)).get_sel_counts();
}

size_t sum(std::vector<uint32_t> const& counts)
{
	return std::accumulate(counts.begin(), counts.end(), size_t(0));
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("columns.csv").toStdString();
	const std::string format = dir.filePath("columns.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (PVRow i = 0; i < BUNDLE; ++i) {
			out << 250000 << "," << 750000 << "\n";
		}
		for (PVRow i = 0; i < SCATTERED; ++i) {
			out << i * 1000 << "," << (i * 7919) % SCATTERED * 1000 << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = *env.root.get_children<Squey::PVView>().front();

	// After TestEnv, which runs an application of its own while it builds. On the
	// processor, which every machine has.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	if (qEnvironmentVariableIsEmpty("FORCE_CPU")) {
		qputenv("FORCE_CPU", "1");
	}
	QApplication app(argc, argv);
	// Declared before the backend, so as to outlive the rendering context.
	std::unique_ptr<PVParallelView::PVFullParallelView> widget;
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVViewRenderingContext& context =
	    *PVParallelView::common::get_rendering_context(view);

	widget = std::make_unique<PVParallelView::PVFullParallelView>();
	auto* scene = new PVParallelView::PVFullParallelScene(widget.get(), view, context,
	                                                      PVParallelView::common::backend());
	widget->setScene(scene);
	// The opacities counted below are those of whole pixels, which antialiasing
	// would spread over the edges of the lines.
	toolbar_action(*widget, "Antialiasing")->setChecked(false);
	scene->first_render();
	widget->resize(800, 500);
	widget->show();
	pump(1000);

	// Opaque, as the lines are unless asked otherwise.
	std::array<size_t, 256> pixels = opacities(*scene);
	PV_ASSERT_VALID(drawn(pixels) > 0 and pixels[255] == drawn(pixels), "opaque pixels",
	                pixels[255], "drawn", drawn(pixels));

	QAction* by_density = toolbar_action(*widget, "Lines by density");
	by_density->setChecked(true);
	pump(1500);

	// The slider starts at its right end, the most opaque it goes.
	const float opacity = scene->line_opacity();
	PV_ASSERT_VALID(std::abs(opacity - std::pow(10.f, -.1f)) < 1e-6f, "line opacity", opacity);

	// Every selected row counted, the bundle on its line.
	std::vector<uint32_t> counts = selected_rows_per_line(context);
	PV_ASSERT_VALID(sum(counts) == BUNDLE + SCATTERED, "rows counted", sum(counts));
	PV_ASSERT_VALID(*std::max_element(counts.begin(), counts.end()) >= BUNDLE,
	                "rows of the bundle's line", *std::max_element(counts.begin(), counts.end()));

	// The bundle stands out; a row on its own has the opacity of one row.
	pixels = opacities(*scene);
	const auto alone = static_cast<size_t>(std::lround(255.f * opacity));
	PV_ASSERT_VALID(alone > 0 and alone < 255, "opacity of a row on its own", alone);
	std::cout << "by density, at " << opacity << ": " << pixels[255] << " opaque pixels, "
	          << pixels[alone] << " of a row's opacity, " << drawn(pixels) << " drawn" << std::endl;
	PV_ASSERT_VALID(pixels[255] > 0, "opaque pixels", pixels[255]);
	PV_ASSERT_VALID(pixels[alone] > pixels[255], "pixels of a row's opacity", pixels[alone],
	                "opaque pixels", pixels[255]);

	// Opaque again once turned off, and the rows no longer counted: a new selection
	// leaves the counts as they were.
	by_density->setChecked(false);
	pump(1500);
	pixels = opacities(*scene);
	PV_ASSERT_VALID(pixels[255] == drawn(pixels), "opaque pixels", pixels[255], "drawn",
	                drawn(pixels));

	Squey::PVSelection half(view.get_row_count());
	half.select_none();
	for (PVRow r = 0; r < view.get_row_count(); r += 2) {
		half.set_bit_fast(r);
	}
	view.set_selection_view(half);
	pump(1500);
	counts = selected_rows_per_line(context);
	PV_ASSERT_VALID(sum(counts) == BUNDLE + SCATTERED, "rows counted while drawn opaque",
	                sum(counts));

	// Turned on again, the new selection is counted.
	by_density->setChecked(true);
	pump(1500);
	counts = selected_rows_per_line(context);
	PV_ASSERT_VALID(sum(counts) == half.bit_count(), "rows counted", sum(counts), "selected",
	                half.bit_count());

	return 0;
}
