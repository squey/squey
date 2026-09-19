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

// A parallel coordinates view only draws the zones it shows. As fewer of them are
// shown -- the window narrows, a dialog covers part of it -- the images of the
// others are dropped, and with them the handles on their renderings, which carry
// on regardless and report to the view once done. Draining the view before it is
// deleted must wait for those too, or they report to a deleted view.
//
// Here the preprocessing of every zone but the first is held at a gate, so that
// their renderings are still running when the view is drained, however fast the
// machine: a rendering that the drain missed reports to the receiver afterwards.

#include "lines_view_dropped_renderings.h"

#include <pvparallelview/PVBCIDrawingBackendQPainter.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZonesProcessor.h>

#include <pvkernel/core/squey_assert.h>

#include <squey/PVView.h>

#include <QCoreApplication>
#include <QTemporaryDir>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common.h"

void RenderingsReceiver::zr_bg_finished(PVParallelView::PVZoneRendering_p, PVZoneID)
{
	++_reports;
}

void RenderingsReceiver::zr_sel_finished(PVParallelView::PVZoneRendering_p, PVZoneID)
{
	++_reports;
}

namespace
{

// Six columns, hence five zones: enough to leave several of them to drop, few
// enough to stay quick in the slowest build.
constexpr size_t COLUMNS = 6;
constexpr size_t ROWS = 1000;

void write_input(std::string const& csv_path, std::string const& format_path)
{
	std::ofstream csv(csv_path);
	for (size_t row = 0; row < ROWS; ++row) {
		for (size_t col = 0; col < COLUMNS; ++col) {
			csv << (col > 0 ? "," : "") << (row * (col + 1)) % 997;
		}
		csv << "\n";
	}

	std::ofstream format(format_path);
	format << "<?xml version='1.0' encoding='UTF-8'?>\n"
	          "<!DOCTYPE PVParamXml>\n"
	          "<param version=\"5\" first_line=\"0\">\n"
	          " <splitter type=\"csv\" sep=\",\" quote=\"&quot;\">\n";
	for (size_t col = 0; col < COLUMNS; ++col) {
		format << "  <field>\n"
		          "   <axis titlecolor=\"#ff921d\" type=\"integer\" group=\"\" tag=\"\" "
		          "color=\"#ffffff\" name=\"column"
		       << col
		       << "\" mapping=\"default\" key=\"false\" plotting=\"default\">\n"
		          "    <mapping mode=\"default\"/>\n"
		          "    <plotting mode=\"default\"/>\n"
		          "   </axis>\n"
		          "  </field>\n";
	}
	format << " </splitter>\n"
	          "</param>\n";
}

// Holds the preprocessing of every zone but one until it is opened.
class Gate
{
  public:
	explicit Gate(PVZoneID open_zone) : _open_zone(open_zone) {}

	void pass(PVZoneID zone_id)
	{
		// Both columns compared: PVZoneID::operator== only looks at the first one.
		if (zone_id.first == _open_zone.first and zone_id.second == _open_zone.second) {
			return;
		}
		std::unique_lock<std::mutex> lock(_mutex);
		_opened.wait(lock, [this]() { return _open; });
	}

	void open()
	{
		{
			std::lock_guard<std::mutex> lock(_mutex);
			_open = true;
		}
		_opened.notify_all();
	}

  private:
	const PVZoneID _open_zone;
	std::mutex _mutex;
	std::condition_variable _opened;
	bool _open = false;
};

// Draws every zone, then the first one alone, and drains the lines view, by
// asking for it or by deleting the view. Returns how many renderings reported
// after the drain was over.
int reports_after_drain(Squey::PVView& view,
                        PVParallelView::PVZonesManager& zm,
                        bool drain_by_deleting)
{
	auto& backend = PVParallelView::PVBCIDrawingBackendQPainter::get();
	PVParallelView::PVRenderingPipeline pipeline(backend);

	Gate gate(zm.get_zone_id(0));
	Squey::PVSelection const& sel = view.get_real_output_selection();
	PVCore::PVHSVColor const* colors = view.get_output_layer_color_buffer();
	PVParallelView::PVZonesProcessor processor_sel = pipeline.declare_processor(
	    [&](PVZoneID zone_id) {
		    gate.pass(zone_id);
		    zm.filter_zone_by_sel(zone_id, sel);
	    },
	    colors, zm);
	PVParallelView::PVZonesProcessor processor_bg = pipeline.declare_processor(
	    [&](PVZoneID zone_id) {
		    gate.pass(zone_id);
		    zm.filter_zone_by_sel_background(zone_id, sel);
	    },
	    colors, zm);

	RenderingsReceiver receiver;
	auto lines_view = std::make_unique<PVParallelView::PVLinesView>(backend, zm, processor_sel,
	                                                                processor_bg, &receiver);

	const size_t zones_count = zm.get_number_of_axes_comb_zones();
	lines_view->render_all_zones_images(0, zones_count * PVParallelView::ZoneMaxWidth, 1.0f);
	PV_VALID(lines_view->get_number_of_visible_zones(), zones_count);
	// The images of all the zones but the first are dropped, while their
	// renderings are held at the gate.
	lines_view->render_all_zones_images(0, 1, 1.0f);
	PV_VALID(lines_view->get_number_of_visible_zones(), size_t(1));

	// A drain that waits for the held renderings cannot be over before they are:
	// the gate opens as soon as the drain is back, or after a while.
	std::atomic<bool> drained = false;
	std::thread opener([&]() {
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
		while (not drained and std::chrono::steady_clock::now() < deadline) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		gate.open();
	});

	if (drain_by_deleting) {
		lines_view.reset();
	} else {
		lines_view->cancel_and_wait_all_rendering();
	}
	// What has reported so far did so before the drain was over.
	QCoreApplication::sendPostedEvents();
	const int reports_before_drain = receiver.reports();
	drained = true;
	opener.join();

	// Whatever still runs goes on to its end, waited for from a thread of its
	// own: the thread of the receiver must not run the ends of the renderings,
	// as it would when waiting for the pipeline itself.
	std::thread([&]() { pipeline.wait_for_all(); }).join();
	QCoreApplication::sendPostedEvents();
	return receiver.reports() - reports_before_drain;
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());
	const std::string csv_path = dir.filePath("columns.csv").toStdString();
	const std::string format_path = csv_path + ".format";
	write_input(csv_path, format_path);

	// Builds the view under a QCoreApplication of its own, gone once it is done.
	TestEnv env(csv_path, format_path);
	QCoreApplication app(argc, argv);
	qRegisterMetaType<PVParallelView::PVZoneRendering_p>();
	qRegisterMetaType<PVZoneID>();

	Squey::PVView& view = *env.root.current_view();
	PVParallelView::PVZonesManager zm(view);
	PV_VALID(zm.get_number_of_axes_comb_zones(), COLUMNS - 1);

	// As a scene does before it is deleted.
	PV_VALID(reports_after_drain(view, zm, false), 0);
	// As when the user closes the view, which deletes it straight away.
	PV_VALID(reports_after_drain(view, zm, true), 0);

	return 0;
}
