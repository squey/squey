//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The scatter thumbnails gallery scrolls by whole rows of thumbnails, a row per notch of a wheel
// whatever their size, and as far as the fingers go on a touchpad. Left to the list it is made of,
// it would scroll a page per notch, and three rows for the few pixels a touchpad counts as one.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVScatterThumbnailsModel.h>
#include <pvparallelview/PVScatterThumbnailsView.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QListView>
#include <QScrollBar>
#include <QSlider>
#include <QTemporaryDir>
#include <QTest>
#include <QWheelEvent>

#include <fstream>
#include <string>

#include "common.h"

namespace
{

constexpr int COLUMN_COUNT = 16;

// Sends @p widget the wheel event of a turn of @p angle eighths of a degree and of a move of
// @p pixels: a wheel tells pixels too under Wayland, and a touchpad the phase of its gesture.
void wheel(QWidget& widget, int angle, int pixels = 0, Qt::ScrollPhase phase = Qt::NoScrollPhase)
{
	const QPointF center = QRectF(widget.rect()).center();
	QWheelEvent event(center, widget.mapToGlobal(center), QPoint(0, pixels), QPoint(0, angle),
	                  Qt::NoButton, Qt::NoModifier, phase, false);
	QCoreApplication::sendEvent(&widget, &event);
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());
	const std::string csv = dir.filePath("columns.csv").toStdString();
	const std::string format = dir.filePath("columns.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (int row = 0; row < 100; row++) {
			for (int col = 0; col < COLUMN_COUNT; col++) {
				out << (col == 0 ? "" : ",") << row * (col + 1);
			}
			out << '\n';
		}
		std::ofstream out_format(format);
		out_format << "<?xml version='1.0' encoding='UTF-8'?>\n<!DOCTYPE PVParamXml>\n"
		              "<param version=\"11\" first_line=\"0\">\n"
		              " <splitter type=\"csv\" sep=\",\">\n";
		for (int col = 0; col < COLUMN_COUNT; col++) {
			out_format << "  <field><axis name=\"c" << col
			           << "\" type=\"number_uint32\"/></field>\n";
		}
		out_format << " </splitter>\n</param>\n";
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();

	// After TestEnv, which runs an application of its own while it builds.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	QApplication app(argc, argv);
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVScatterThumbnailsView gallery(*view);
	gallery.resize(900, 600);
	gallery.show();
	PV_ASSERT_VALID(QTest::qWaitForWindowExposed(&gallery));

	auto* list = gallery.findChild<QListView*>();
	PV_ASSERT_VALID(list != nullptr);
	QScrollBar* bar = list->verticalScrollBar();
	const QAbstractItemModel& pairs = *list->model();
	// From one row of thumbnails to the next, as the list lays them out.
	const auto row_height = [&]() {
		const auto top = [&](int item) { return list->visualRect(pairs.index(item, 0)).top(); };
		for (int item = 1; item < pairs.rowCount(); item++) {
			if (top(item) != top(0)) {
				return top(item) - top(0);
			}
		}
		return 0;
	};
	// Rows enough to scroll a few of them.
	PV_ASSERT_VALID(QTest::qWaitFor(
	    [&]() { return row_height() > 0 and bar->maximum() > 3 * row_height(); }, 10000));
	const int row = row_height();
	PV_ASSERT_VALID(row < bar->pageStep(), "row", row, "page", bar->pageStep());

	// A row per notch of a wheel, rather than a page, and only once the eighths of a notch a
	// high-resolution wheel reports add up to one.
	wheel(*list->viewport(), -120);
	PV_VALID(bar->value(), row);
	for (int eighth = 1; eighth < 8; eighth++) {
		wheel(*list->viewport(), -15);
		PV_VALID(bar->value(), row);
	}
	wheel(*list->viewport(), -15);
	PV_VALID(bar->value(), 2 * row);

	// From between two rows, to the next one either way.
	bar->setValue(2 * row + row / 2);
	wheel(*list->viewport(), -120);
	PV_VALID(bar->value(), 3 * row);
	bar->setValue(2 * row + row / 2);
	wheel(*list->viewport(), 120);
	PV_VALID(bar->value(), 2 * row);

	// The few pixels a wheel tells under Wayland do not make it a touchpad.
	bar->setValue(0);
	wheel(*list->viewport(), -120, -15);
	PV_VALID(bar->value(), row);

	// Of larger thumbnails, as large a row.
	gallery.findChild<QSlider*>()->setValue(PVParallelView::PVScatterThumbnailsModel::max_size);
	PV_ASSERT_VALID(QTest::qWaitFor([&]() { return row_height() > row; }, 10000));
	bar->setValue(0);
	wheel(*list->viewport(), -120);
	PV_VALID(bar->value(), row_height());

	// The pixels of a touchpad, and back as the fingers lift.
	bar->setValue(0);
	wheel(*list->viewport(), -120, -10, Qt::ScrollUpdate);
	PV_VALID(bar->value(), 10);
	wheel(*list->viewport(), 24, 2, Qt::ScrollUpdate);
	PV_VALID(bar->value(), 8);

	return 0;
}
