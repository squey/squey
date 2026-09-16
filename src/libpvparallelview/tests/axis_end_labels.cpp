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

// The values printed at the ends of an axis are the ones drawn there.
//
// The parallel view draws a column's smallest position at the top of its axis --
// positions are stored inverted, so that is its largest value -- and prints, when
// asked, the value found at each end. From 2016 on each end carried the other end's
// value. The fault was in neither half alone but in how they were matched, so both
// are looked at here: the labels, and the picture they describe.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVViewRenderingContext.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QElapsedTimer>
#include <QGraphicsTextItem>
#include <QImage>
#include <QKeyEvent>
#include <QTemporaryDir>
#include <QThread>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"

namespace
{

/**
 * A column whose top is held by one row: 999 on row 0, and a digit on every other
 * row. Under the default scaling the digits share the bottom hundredth of the
 * axis, so the only line reaching the top of it is row 0's, and none reaches its
 * middle. It is the second axis, and the lines are looked at as they arrive: the
 * view scrolls to fit, and the first axis may sit left of the picture.
 */
constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="spread" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="peak" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

bool lit(QRgb pixel, QRgb background)
{
	return std::abs(qRed(pixel) - qRed(background)) + std::abs(qGreen(pixel) - qGreen(background)) +
	           std::abs(qBlue(pixel) - qBlue(background)) >
	       60;
}

//! How many lit pixels a band of rows holds, just left of an axis.
int lit_in_band(const QImage& image, int x, int top, int bottom, QRgb background)
{
	int count = 0;
	for (int y = std::max(top, 0); y <= std::min(bottom, image.height() - 1); ++y) {
		// Clipped: a pixel read outside the picture comes back black, which is not
		// the background and would count as a line.
		for (int column = std::max(x - 12, 0); column < std::min(x - 4, image.width()); ++column) {
			count += lit(image.pixel(column, y), background);
		}
	}
	return count;
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("peak.csv").toStdString();
	const std::string format = dir.filePath("peak.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (int i = 0; i < 1000; ++i) {
			out << i % 7 << "," << (i == 0 ? 999 : i % 10) << "\n";
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

	// Built as the display builds it.
	PVParallelView::PVViewRenderingContext* context =
	    PVParallelView::common::get_rendering_context(view);
	auto* widget = new PVParallelView::PVFullParallelView();
	auto* scene = new PVParallelView::PVFullParallelScene(widget, view, *context,
	                                                      PVParallelView::common::backend());
	widget->setScene(scene);
	scene->first_render();
	widget->resize(1400, 900);
	widget->show();

	// Y shows the values at the ends of every axis.
	QKeyEvent press(QEvent::KeyPress, Qt::Key_Y, Qt::NoModifier);
	QApplication::sendEvent(scene, &press);

	PVParallelView::PVAxisGraphicsItem* axis = nullptr;
	for (QGraphicsItem* item : scene->items()) {
		auto* candidate = dynamic_cast<PVParallelView::PVAxisGraphicsItem*>(item);
		if (candidate != nullptr and candidate->get_original_axis_column() == PVCol(1)) {
			axis = candidate;
		}
	}
	PV_ASSERT_VALID(axis != nullptr, "the second axis", "is not in the scene");

	// The four values of the second axis, from the top of the picture down: the
	// axis's own and the layer's above the axis, then the same two below it.
	std::vector<QGraphicsTextItem*> values;
	for (QGraphicsItem* child : axis->childItems()) {
		if (auto* text = dynamic_cast<QGraphicsTextItem*>(child)) {
			values.push_back(text);
		}
	}
	PV_ASSERT_VALID(values.size() == 4, "values at the ends of an axis", values.size());
	const auto top_down = [&values] {
		std::sort(values.begin(), values.end(), [](QGraphicsTextItem* a, QGraphicsTextItem* b) {
			return a->scenePos().y() < b->scenePos().y();
		});
	};

	// Where the axis runs in the picture, from its position and from the label placed
	// just past its bottom end. Read again on every pass: the view lays itself out
	// while the loop runs.
	int axis_x = 0;
	int top = 0;
	int bottom = 0;
	const auto locate = [&] {
		top_down();
		const qreal axis_top = axis->mapToScene(QPointF(0, 0)).y();
		const qreal axis_bottom =
		    values[2]->scenePos().y() - PVParallelView::PVAxisGraphicsItem::axis_extend;
		const QPoint origin = widget->mapFromScene(axis->mapToScene(QPointF(0, 0)));
		axis_x = origin.x();
		top = widget->mapFromScene(QPointF(0, axis_top)).y();
		bottom = widget->mapFromScene(QPointF(0, axis_bottom)).y();
	};

	// The zones are drawn on a background pool and posted back, so the event loop is
	// run until the picture has lines in it -- waiting, not only pumping: pumping
	// alone returns as soon as nothing is queued, before anything has been drawn.
	QImage picture;
	QRgb background = 0;
	const auto look = [&] {
		QApplication::processEvents(QEventLoop::AllEvents, 10);
		QThread::msleep(10);
		locate();
		picture = widget->viewport()->grab().toImage();
		background = picture.pixel(picture.width() / 2, picture.height() - 1);
	};
	QElapsedTimer waited;
	waited.start();
	int drawn = 0;
	while (drawn == 0 and waited.elapsed() < 30000) {
		look();
		// Along the axis only: the status line above it is text on the same ground.
		drawn = lit_in_band(picture, axis_x, top + 1, bottom - 1, background);
	}
	PV_ASSERT_VALID(drawn > 0, "no line was drawn within ms", waited.elapsed());
	// And a moment more, for whatever was still on its way.
	for (int i = 0; i < 30; ++i) {
		look();
	}

	std::cout << "labels, top down:";
	for (QGraphicsTextItem* value : values) {
		std::cout << " " << value->toPlainText().toStdString();
	}
	std::cout << std::endl;

	const int span = bottom - top;
	PV_ASSERT_VALID(span > 100, "axis length in pixels", span);

	// The picture: row 0's line alone reaches the top of the axis, the digits reach
	// its bottom, and nothing reaches the middle. Checked before the labels, so that
	// a failure says which half moved; a pixel short of each end, clear of the labels
	// printed just beyond it.
	const int at_top = lit_in_band(picture, axis_x, top + 1, top + span / 50, background);
	const int in_middle =
	    lit_in_band(picture, axis_x, top + span / 4, bottom - span / 4, background);
	const int at_bottom = lit_in_band(picture, axis_x, bottom - span / 50, bottom - 1, background);
	std::cout << "lit pixels next to the axis: top " << at_top << ", middle " << in_middle
	          << ", bottom " << at_bottom << std::endl;
	PV_ASSERT_VALID(at_top > 0, "the largest value", "is not drawn at the top");
	PV_ASSERT_VALID(in_middle == 0, "lit pixels mid-axis", in_middle);
	PV_ASSERT_VALID(at_bottom > 0, "the small values", "are not drawn at the bottom");

	// And the labels say so.
	for (size_t i = 0; i < 2; ++i) {
		PV_ASSERT_VALID(values[i]->toPlainText() == "999", "label above the axis",
		                values[i]->toPlainText().toStdString());
	}
	for (size_t i = 2; i < 4; ++i) {
		PV_ASSERT_VALID(values[i]->toPlainText() == "0", "label below the axis",
		                values[i]->toPlainText().toStdString());
	}

	delete widget;
	return 0;
}
