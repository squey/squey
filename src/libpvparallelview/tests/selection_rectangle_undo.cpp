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

// Steps brought back with the selection rectangle that drew them.
//
// A step keeps the rectangle in the coordinates of the scene, which only mean
// something on the axes as they were scaled when it was drawn. Once the axes have
// been stretched over a selection, the same edges frame other rows, and the
// rectangle must not come back.
//
// Under a stretch on each selection, every step lands on axes rescaled since its
// rectangle was drawn, and the selection being restored sets off a stretch of its
// own, which drops whatever rectangle is shown: one brought back would blink on
// screen for the time that stretch waits.
//
// Without a stretch, the rectangle comes back where it was drawn, and stays.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSelectionRectangleItem.h>
#include <pvparallelview/PVViewRenderingContext.h>

#include <squey/PVAnalysisHistory.h>
#include <squey/PVRoot.h>
#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QGraphicsSceneMouseEvent>
#include <QTemporaryDir>
#include <QThread>

#include <atomic>
#include <fstream>
#include <iostream>
#include <string>

#include "common.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="rising" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="scattered" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr PVRow ROWS = 1000;

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

bool shown(PVParallelView::PVSelectionRectangleItem const& item)
{
	return item.isVisible() and not item.get_rect().isNull();
}

/**
 * What the rectangle looked like while the event loop ran, sampled often enough
 * to catch a rectangle shown for the 400 ms a stretch waits.
 */
struct Watched {
	bool ever_shown = false;
	bool ever_hidden = false;
	bool ever_elsewhere = false;
};

Watched watch(PVParallelView::PVSelectionRectangleItem const& item,
              int ms,
              QRectF const& expected = QRectF())
{
	Watched seen;
	QElapsedTimer waited;
	waited.start();
	do {
		if (shown(item)) {
			seen.ever_shown = true;
			seen.ever_elsewhere |= item.get_rect() != expected;
		} else {
			seen.ever_hidden = true;
		}
		QApplication::processEvents(QEventLoop::AllEvents, 5);
		QThread::msleep(2);
	} while (waited.elapsed() < ms);
	return seen;
}

/**
 * Run the event loop until the scaling has changed since it read `before`, which
 * is how a stretch is known to have gone through. Watches the rectangle the while,
 * and a little after.
 */
Watched watch_until_rescaled(PVParallelView::PVSelectionRectangleItem const& item,
                             std::atomic<int> const& rescalings,
                             int before)
{
	Watched seen;
	QElapsedTimer waited;
	waited.start();
	while (rescalings == before and waited.elapsed() < 10000) {
		seen.ever_shown |= shown(item);
		QApplication::processEvents(QEventLoop::AllEvents, 5);
		QThread::msleep(2);
	}
	PV_ASSERT_VALID(rescalings != before, "a stretch", "never came");

	const Watched after = watch(item, 300);
	seen.ever_shown |= after.ever_shown;
	return seen;
}

void send_mouse(QGraphicsScene& scene, QEvent::Type type, QPointF p, Qt::MouseButtons buttons)
{
	QGraphicsSceneMouseEvent event(type);
	event.setScenePos(p);
	event.setButton(type == QEvent::GraphicsSceneMouseMove ? Qt::NoButton : Qt::LeftButton);
	event.setButtons(buttons);
	event.setAccepted(false);
	QApplication::sendEvent(&scene, &event);
}

/**
 * Draw a rectangle the way the mouse does, which is what opens the step and hands
 * it the rectangle.
 */
void draw(QGraphicsScene& scene, QPointF from, QPointF to)
{
	send_mouse(scene, QEvent::GraphicsSceneMousePress, from, Qt::LeftButton);
	for (int i = 1; i <= 10; ++i) {
		send_mouse(scene, QEvent::GraphicsSceneMouseMove, from + (to - from) * (i / 10.),
		           Qt::LeftButton);
		QApplication::processEvents();
	}
	send_mouse(scene, QEvent::GraphicsSceneMouseRelease, to, Qt::NoButton);
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
		for (PVRow i = 0; i < ROWS; ++i) {
			out << i << "," << (i * 7) % ROWS << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = *env.root.get_children<Squey::PVView>().front();
	Squey::PVAnalysisHistory& history = env.root.history();

	// After TestEnv, which runs an application of its own while it builds. On the
	// processor, which every machine has.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	if (qEnvironmentVariableIsEmpty("FORCE_CPU")) {
		qputenv("FORCE_CPU", "1");
	}
	QApplication app(argc, argv);
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVViewRenderingContext& context =
	    *PVParallelView::common::get_rendering_context(view);

	auto* widget = new PVParallelView::PVFullParallelView();
	auto* scene = new PVParallelView::PVFullParallelScene(widget, view, context,
	                                                      PVParallelView::common::backend());
	widget->setScene(scene);
	scene->first_render();
	widget->resize(800, 500);
	widget->show();
	pump(500);

	PVParallelView::PVSelectionRectangleItem* item = nullptr;
	for (QGraphicsItem* i : scene->items()) {
		if (auto* r = dynamic_cast<PVParallelView::PVSelectionRectangleItem*>(i)) {
			item = r;
		}
	}
	PV_ASSERT_VALID(item != nullptr, "the selection rectangle", "is not in the scene");

	// Told from the thread the scaling is computed in.
	std::atomic<int> rescalings{0};
	sigc::connection counting = view.get_parent<Squey::PVScaled>()._scaled_updated.connect(
	    [&rescalings](QList<PVCol> const& columns) {
		    if (not columns.empty()) {
			    ++rescalings;
		    }
	    });

	// Both rectangles inside the zone between the two axes, the second one well
	// clear of the first's handles: pressed on one of those, the mouse would move
	// the first rectangle instead of drawing another.
	auto const& lines_view = scene->get_lines_view();
	const double left = lines_view.get_left_border_position_of_zone_in_scene(0);
	const double width = lines_view.get_zone_width(0);
	const QPointF first_from(left + width * 0.2, 20);
	const QPointF first_to(left + width * 0.8, 200);
	const QPointF second_from(left + width * 0.1, 300);
	const QPointF second_to(left + width * 0.5, 400);

	// ------------------------------------------------------------ without a stretch

	draw(*scene, first_from, first_to);
	pump(200);
	PV_ASSERT_VALID(shown(*item), "the first rectangle", "is not shown");
	const QRectF first = item->get_rect();

	draw(*scene, second_from, second_to);
	pump(200);
	PV_ASSERT_VALID(shown(*item), "the second rectangle", "is not shown");
	const QRectF second = item->get_rect();
	PV_VALID(history.size(), size_t(3));

	history.undo();
	Watched seen = watch(*item, 1000, first);
	PV_ASSERT_VALID(not seen.ever_hidden and not seen.ever_elsewhere, "why",
	                "a step comes back with the rectangle that drew it, where it was drawn");

	history.redo();
	seen = watch(*item, 1000, second);
	PV_ASSERT_VALID(not seen.ever_hidden and not seen.ever_elsewhere);

	// ----------------------------------------------------------- stretch on demand

	history.clear();

	draw(*scene, first_from, first_to);
	pump(200);
	PV_ASSERT_VALID(shown(*item), "the first rectangle", "is not shown");

	int before = rescalings;
	scene->rescale_on_selection();
	PV_ASSERT_VALID(rescalings != before, "the stretch", "moved no axis");
	PV_ASSERT_VALID(not shown(*item), "why", "a stretch drops the rectangle it moved the axes of");

	draw(*scene, second_from, second_to);
	pump(200);
	PV_ASSERT_VALID(shown(*item), "the second rectangle", "is not shown");
	const QRectF stretched = item->get_rect();

	history.undo();
	seen = watch(*item, 1000);
	PV_ASSERT_VALID(not seen.ever_shown, "why",
	                "a rectangle drawn before the axes were stretched frames other rows now");

	history.redo();
	seen = watch(*item, 1000, stretched);
	PV_ASSERT_VALID(not seen.ever_hidden and not seen.ever_elsewhere, "why",
	                "one drawn on the axes as they still are comes back");

	// ------------------------------------------------------ stretch on each selection

	history.clear();
	scene->set_auto_scale_on_selection(true);

	before = rescalings;
	draw(*scene, first_from, first_to);
	PV_ASSERT_VALID(shown(*item), "the first rectangle", "is not shown");
	watch_until_rescaled(*item, rescalings, before);
	PV_ASSERT_VALID(not shown(*item), "why", "the stretch drops the rectangle");

	before = rescalings;
	draw(*scene, second_from, second_to);
	PV_ASSERT_VALID(shown(*item), "the second rectangle", "is not shown");
	watch_until_rescaled(*item, rescalings, before);
	PV_ASSERT_VALID(not shown(*item), "why", "the stretch drops the rectangle");

	const auto step_back = [&](const char* where) {
		const int rescalings_before = rescalings;
		const size_t position = history.position();
		if (std::string(where) == "undo") {
			history.undo();
		} else {
			history.redo();
		}
		PV_ASSERT_VALID(history.position() != position, where, "went nowhere");

		// Landing on a rectangle's step sets off a stretch over the selection it
		// restores, which moves the axes: waited for, so that nothing shown before
		// it goes unseen. The start of the trail carries no rectangle, and the
		// stretch over its selection need not move them.
		const Watched landed = history.position() == 0
		                           ? watch(*item, 1000)
		                           : watch_until_rescaled(*item, rescalings, rescalings_before);
		PV_ASSERT_VALID(not landed.ever_shown, "why",
		                "a step lands on axes rescaled since its rectangle was drawn", "after",
		                where, "at step", history.position());
	};
	step_back("undo");
	step_back("undo");
	step_back("redo");
	step_back("redo");

	counting.disconnect();
	delete widget;
	return 0;
}
