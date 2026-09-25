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

// A zoomed parallel view while the axes under it are scaled again.
//
// A new scaling reaches the views from the thread a progress box runs it in, and
// the zoomed view rebuilt its axis menu from there. The menu's signals then came
// back to the GUI thread queued, past the blocking meant to silence them, and each
// asked the view to switch to the axis it named: the rows the menu goes through as
// it is refilled, and -- since the menu took to saying every change twice, the
// second time with no axis at all -- an axis numbered -1, or none. A switch
// deletes the view's own zoom sliders, which the view could take for a request to
// close: with a dock around it the view vanished, without one it crashed.
//
// So the view is zoomed in, every column is scaled again under a progress box, as
// the axis menu does it, and the zoomed view has to still be there, on its axis,
// zoomed as it was. Picking another axis in its menu switches it once, to that axis -- here
// one whose column is numbered like the axis left, which is what the view compared
// its own sliders' removal with. And a menu emptied, naming no axis, leaves the
// view where it is.

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVViewRenderingContext.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>

#include <squey/PVScaled.h>
#include <squey/PVView.h>
#include <squey/widgets/PVAxisComboBox.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QGraphicsSceneWheelEvent>
#include <QScrollBar>
#include <QTemporaryDir>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

#include <fstream>
#include <iostream>
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
   <axis name="falling" type="number_uint32">
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
	const std::string csv = dir.filePath("columns.csv").toStdString();
	const std::string format = dir.filePath("columns.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (PVRow i = 0; i < ROWS; ++i) {
			out << i << "," << (i * 7919) % ROWS << "," << ROWS - 1 - i << "," << i / 100
			    << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = *env.root.get_children<Squey::PVView>().front();
	// The first two columns swapped: the second axis shows column 0, the first column 1.
	view.set_axes_combination({PVCol(1), PVCol(0), PVCol(2), PVCol(3)});

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

	// The parallel view, whose zones the rescaling rebuilds as well.
	auto* full_view = new PVParallelView::PVFullParallelView();
	auto* full_scene = new PVParallelView::PVFullParallelScene(full_view, view, context,
	                                                           PVParallelView::common::backend());
	full_view->setScene(full_scene);
	full_scene->first_render();
	full_view->resize(800, 500);
	full_view->show();

	// The zoomed view on the second axis, in a widget standing for its dock: a view
	// that closes itself closes what holds it.
	QWidget dock;
	auto* layout = new QVBoxLayout(&dock);
	auto* zoomed_view = new PVParallelView::PVZoomedParallelView(view.get_axes_combination(), &dock);
	auto* zoomed_scene =
	    new PVParallelView::PVZoomedParallelScene(zoomed_view, view, context, PVCombCol(1));
	zoomed_view->set_scene(zoomed_scene);
	layout->addWidget(zoomed_view);
	dock.resize(600, 700);
	dock.show();
	pump(800);

	// Every axis switch the view is asked for, from its menu.
	std::vector<PVCombCol> switches;
	auto* params = zoomed_view->findChild<PVParallelView::PVZoomedParallelViewParamsWidget*>();
	PV_ASSERT_VALID(params != nullptr, "the zoomed view's menu", "was not found");
	QObject::connect(params, &PVParallelView::PVZoomedParallelViewParamsWidget::change_to_col, &app,
	                 [&switches](PVCombCol axis) { switches.push_back(axis); });

	// Zoomed in, which a switch would undo.
	QGraphicsSceneWheelEvent wheel(QEvent::GraphicsSceneWheel);
	wheel.setDelta(120 * 10);
	wheel.setModifiers(Qt::NoModifier);
	QApplication::sendEvent(zoomed_scene, &wheel);
	pump(800);
	const int zoomed_range = zoomed_view->get_vertical_scrollbar()->maximum();
	PV_ASSERT_VALID(zoomed_range > 0, "scroll range once zoomed in", zoomed_range);

	// Every column scaled again, in the thread of a progress box, as the axis menu has
	// a new scaling computed.
	Squey::PVScaled& scaled = view.get_parent<Squey::PVScaled>();
	for (PVCol col(0); col < PVCol(4); col++) {
		scaled.invalidate_column(col);
	}
	PVCore::PVProgressBox::progress([&scaled](PVCore::PVProgressBox&) { scaled.update_scaling(); },
	                                QObject::tr("Updating scaling..."), nullptr);
	pump(1500);

	std::cout << "after rescaling: " << switches.size() << " switches asked, axis "
	          << zoomed_scene->get_axis_index().value() << ", scroll range "
	          << zoomed_view->get_vertical_scrollbar()->maximum() << " (was " << zoomed_range
	          << "), dock " << (dock.isVisible() ? "open" : "closed") << std::endl;
	PV_ASSERT_VALID(dock.isVisible(), "the zoomed view", "was closed by a rescaling");
	PV_ASSERT_VALID(switches.empty(), "axis switches asked by a rescaling", switches.size());
	PV_ASSERT_VALID(zoomed_scene->get_axis_index() == PVCombCol(1), "axis after a rescaling",
	                zoomed_scene->get_axis_index().value());
	PV_ASSERT_VALID(zoomed_view->get_vertical_scrollbar()->maximum() == zoomed_range,
	                "scroll range after a rescaling",
	                zoomed_view->get_vertical_scrollbar()->maximum());

	// Picking the first axis, which shows column 1 -- the number of the axis left --
	// switches once, to that one.
	auto* menu = params->findChild<PVWidgets::PVAxisComboBox*>();
	PV_ASSERT_VALID(menu != nullptr, "the axis menu", "was not found");
	menu->setCurrentIndex(0);
	pump(500);
	std::cout << "after picking the first axis: " << switches.size() << " switches asked";
	for (PVCombCol axis : switches) {
		std::cout << " " << axis.value();
	}
	std::cout << ", dock " << (dock.isVisible() ? "open" : "closed") << std::endl;
	PV_ASSERT_VALID(dock.isVisible(), "the zoomed view", "was closed by a pick");
	PV_ASSERT_VALID(switches.size() == 1, "axis switches asked by one pick", switches.size());
	PV_ASSERT_VALID(switches.front() == PVCombCol(0), "axis asked", switches.front().value());
	PV_ASSERT_VALID(zoomed_scene->get_axis_index() == PVCombCol(0), "axis after the pick",
	                zoomed_scene->get_axis_index().value());

	// A menu emptied names no axis, and the view stays where it is.
	switches.clear();
	menu->clear();
	pump(500);
	PV_ASSERT_VALID(switches.size() == 1 and switches.front() == PVCombCol(),
	                "switches asked by an emptied menu", switches.size());
	PV_ASSERT_VALID(zoomed_scene->get_axis_index() == PVCombCol(0), "axis after emptying",
	                zoomed_scene->get_axis_index().value());
	PV_ASSERT_VALID(dock.isVisible(), "the zoomed view", "was closed by an emptied menu");

	delete full_view;
	return 0;
}
