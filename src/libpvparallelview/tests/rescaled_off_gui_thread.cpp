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

// Views told of a new scaling from the thread it was computed in.
//
// A scaling asked for from the GUI is computed under a progress box, in a thread of
// its own, and its listeners are told from there. Whatever they do to widgets has
// to happen in the GUI thread, which it did not: the parallel, zoomed and scatter
// views disabled and enabled themselves from that thread, and the series view
// opened its sampling progress box from it.
//
// Every one of them is open while a column is given another scaling mode, the way
// the axis menu does it. What reaches a widget from another thread is counted, as
// are the warnings Qt prints about threads. And the work is still done, from the
// right thread: every view is enabled again.

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZoomConverter.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVSeriesViewWidget.h>
#include <pvparallelview/PVViewRenderingContext.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <squey/PVScaled.h>
#include <squey/PVView.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="when" type="time" type_format="epoch">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
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
  <field>
   <axis name="stepped" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr PVRow ROWS = 5000;

//! What happened away from the GUI thread, for the report.
std::mutex g_mutex;
std::vector<std::string> g_off_thread;

void off_thread(const std::string& what)
{
	const std::lock_guard<std::mutex> lock(g_mutex);
	g_off_thread.push_back(what);
}

QtMessageHandler g_previous_handler = nullptr;

void on_message(QtMsgType type, const QMessageLogContext& context, const QString& message)
{
	if (message.contains("thread", Qt::CaseInsensitive)) {
		off_thread("Qt: " + message.toStdString());
	}
	if (g_previous_handler != nullptr) {
		g_previous_handler(type, context, message);
	}
}

/**
 * Counts the events a widget receives from another thread than its own.
 *
 * An event sent is delivered, event filters first, in the thread that sends it --
 * which is what makes a widget touched from elsewhere visible from here.
 */
class OffThreadEvents : public QObject
{
  public:
	using QObject::QObject;

	bool eventFilter(QObject* watched, QEvent* event) override
	{
		if (QThread::currentThread() != watched->thread()) {
			off_thread(watched->metaObject()->className() + std::string(" received event ") +
			           std::to_string(int(event->type())));
		}
		return false;
	}
};

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
		QApplication::processEvents(QEventLoop::AllEvents, 10);
		QThread::msleep(5);
	}
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("series.csv").toStdString();
	const std::string format = dir.filePath("series.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (PVRow i = 0; i < ROWS; ++i) {
			out << 1000000000 + i * 60 << "," << i << "," << (i * 7919) % ROWS << "," << i / 100
			    << "\n";
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
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVViewRenderingContext& context =
	    *PVParallelView::common::get_rendering_context(view);
	OffThreadEvents watch;

	auto* full_view = new PVParallelView::PVFullParallelView();
	auto* full_scene = new PVParallelView::PVFullParallelScene(full_view, view, context,
	                                                           PVParallelView::common::backend());
	full_view->setScene(full_scene);
	full_scene->first_render();
	full_view->resize(800, 500);
	full_view->show();
	full_view->installEventFilter(&watch);

	// In a widget standing for its dock, as a zoomed view closing itself closes that.
	QWidget dock;
	auto* layout = new QVBoxLayout(&dock);
	auto* zoomed_view = new PVParallelView::PVZoomedParallelView(view.get_axes_combination(), &dock);
	zoomed_view->set_scene(
	    new PVParallelView::PVZoomedParallelScene(zoomed_view, view, context, PVCombCol(1)));
	layout->addWidget(zoomed_view);
	dock.resize(600, 600);
	dock.show();
	zoomed_view->installEventFilter(&watch);

	auto* scatter_view =
	    new PVParallelView::PVScatterView(view, context, PVZoneID{PVCol(1), PVCol(2)});
	scatter_view->resize(600, 500);
	scatter_view->show();
	scatter_view->installEventFilter(&watch);

	// The series view along the time, which samples again whatever it plots.
	auto* series_view = new PVParallelView::PVSeriesViewWidget(&view, PVCol(0));
	series_view->resize(600, 400);
	series_view->show();
	series_view->installEventFilter(&watch);

	pump(2000);
	g_previous_handler = qInstallMessageHandler(on_message);

	// A new mode for the second column, recomputed as the axis menu recomputes it.
	Squey::PVScaled& scaled = view.get_parent<Squey::PVScaled>();
	scaled.get_properties_for_col(PVCol(1)).set_mode("log");
	PVCore::PVProgressBox::progress([&scaled](PVCore::PVProgressBox&) { scaled.update_scaling(); },
	                                QObject::tr("Updating scaling..."), nullptr);
	pump(2000);

	qInstallMessageHandler(g_previous_handler);
	{
		const std::lock_guard<std::mutex> lock(g_mutex);
		for (const std::string& what : g_off_thread) {
			std::cout << "off the GUI thread: " << what << std::endl;
		}
		PV_ASSERT_VALID(g_off_thread.empty(), "things done off the GUI thread",
		                g_off_thread.size());
	}
	PV_ASSERT_VALID(full_view->isEnabled(), "the parallel view", "was left disabled");
	PV_ASSERT_VALID(zoomed_view->isEnabled(), "the zoomed view", "was left disabled");
	PV_ASSERT_VALID(scatter_view->isEnabled(), "the scatter view", "was left disabled");

	full_view->removeEventFilter(&watch);
	zoomed_view->removeEventFilter(&watch);
	scatter_view->removeEventFilter(&watch);
	series_view->removeEventFilter(&watch);
	delete series_view;
	delete scatter_view;
	delete full_view;
	return 0;
}
