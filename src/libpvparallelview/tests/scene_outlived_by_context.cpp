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

// A parallel view closed while the data it showed is still being worked on.
//
// The scaling outlives every parallel view built on the data: one opened in a dock
// is destroyed as the dock is closed. Whatever such a view hands the scaling has to
// go with it. A lambda capturing the scene is not disconnected by sigc::trackable,
// so the next rescaling called into the scene that was gone.
//
// And the densities a rescaling invalidates are those of the axes showing the
// columns it moved, found by column: taken by position in the combination, the
// refresh landed on another axis, or past the last one, as soon as the
// combination was anything but every column in order.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVViewRenderingContext.h>

#include <squey/PVScaled.h>
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>

#include <fstream>
#include <iostream>
#include <map>
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
		QApplication::processEvents(QEventLoop::AllEvents, 10);
		QThread::msleep(5);
	}
}

//! A parallel view on the data, built as the display builds it.
PVParallelView::PVFullParallelView* open_parallel_view(Squey::PVView& view,
                                                       PVParallelView::PVViewRenderingContext& context)
{
	auto* widget = new PVParallelView::PVFullParallelView();
	auto* scene = new PVParallelView::PVFullParallelScene(widget, view, context,
	                                                      PVParallelView::common::backend());
	widget->setScene(scene);
	scene->first_render();
	widget->resize(800, 500);
	widget->show();
	pump(500);
	return widget;
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
			out << i << "," << (i * 7) % ROWS << "," << ROWS - 1 - i << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = *env.root.get_children<Squey::PVView>().front();
	// The last column first and the first last, the middle one left out: an axis's
	// position is no longer the column it shows.
	view.set_axes_combination({PVCol(2), PVCol(0)});

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

	// A view opened and closed, as a dock is.
	delete open_parallel_view(view, context);

	// And one that stays, whose densities are watched, by the column each axis shows.
	auto* widget = open_parallel_view(view, context);
	auto* scene = static_cast<PVParallelView::PVFullParallelScene*>(widget->scene());
	PV_ASSERT_VALID(scene->axes_count() == 2, "axes", scene->axes_count());
	scene->enable_density_on_axes(true);
	std::map<PVCol, int> redrawn;
	for (QGraphicsItem* item : scene->items()) {
		if (auto* axis = dynamic_cast<PVParallelView::PVAxisGraphicsItem*>(item)) {
			const PVCol col = axis->get_original_axis_column();
			QObject::connect(axis, &PVParallelView::PVAxisGraphicsItem::density_changed, &app,
			                 [&redrawn, col] { ++redrawn[col]; });
		}
	}
	scene->update();
	pump(1500);

	// Every view the context knows of hears of a new selection, the closed one included
	// if it left anything behind.
	Squey::PVSelection half(view.get_row_count());
	half.select_none();
	for (PVRow r = 0; r < view.get_row_count(); r += 2) {
		half.set_bit_fast(r);
	}
	redrawn.clear();
	view.set_selection_view(half);
	scene->update();
	pump(1500);
	// Densities are counted over the selection, so every axis redraws its own.
	std::cout << "after selecting, densities redrawn: column 0 " << redrawn[PVCol(0)]
	          << ", column 2 " << redrawn[PVCol(2)] << std::endl;
	PV_ASSERT_VALID(redrawn[PVCol(0)] > 0 and redrawn[PVCol(2)] > 0, "densities redrawn",
	                "after a selection change", "column 0", redrawn[PVCol(0)], "column 2",
	                redrawn[PVCol(2)]);

	// Rescaling one column: the scaling says which columns moved, and only the axis
	// showing the one that did has its density to redraw. Column 0 is shown second.
	Squey::PVScaled& scaled = view.get_parent<Squey::PVScaled>();
	redrawn.clear();
	scaled.invalidate_column(PVCol(0));
	scaled.update_scaling();
	scene->update();
	pump(1500);
	std::cout << "after rescaling column 0, densities redrawn: column 0 " << redrawn[PVCol(0)]
	          << ", column 2 " << redrawn[PVCol(2)] << std::endl;
	PV_ASSERT_VALID(redrawn[PVCol(0)] > 0, "densities redrawn on the axis showing column 0",
	                redrawn[PVCol(0)]);
	PV_ASSERT_VALID(redrawn[PVCol(2)] == 0, "densities redrawn on the axis showing column 2",
	                redrawn[PVCol(2)]);

	// And column 2, shown first but numbered past the last axis.
	redrawn.clear();
	scaled.invalidate_column(PVCol(2));
	scaled.update_scaling();
	scene->update();
	pump(1500);
	std::cout << "after rescaling column 2, densities redrawn: column 0 " << redrawn[PVCol(0)]
	          << ", column 2 " << redrawn[PVCol(2)] << std::endl;
	PV_ASSERT_VALID(redrawn[PVCol(2)] > 0, "densities redrawn on the axis showing column 2",
	                redrawn[PVCol(2)]);
	PV_ASSERT_VALID(redrawn[PVCol(0)] == 0, "densities redrawn on the axis showing column 0",
	                redrawn[PVCol(0)]);

	delete widget;
	return 0;
}
