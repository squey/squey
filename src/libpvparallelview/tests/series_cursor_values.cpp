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

// The list of a series view shows, next to each drawn series, its values under the
// cursor: the value when the column of pixels holds one, their range when it holds
// several, among the selected rows only. Zoomed in, a column without any shows those
// of the closest one within the hole of the crosshair. Nothing once the cursor has left.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesTreeWidget.h>
#include <pvparallelview/PVSeriesViewWidget.h>
#include <pvparallelview/PVSeriesViewZoomer.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QMouseEvent>
#include <QTemporaryDir>
#include <QTest>

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
   <axis name="time" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="power" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

const PVCol POWER(1);

// The first half of the rows holds 7, the second 1 and 9 in turn: enough rows for a column
// of pixels to hold several, however wide the view is.
constexpr PVRow ROW_COUNT = 10000;

QString value_of(const PVSeriesTreeModel& model, PVCol series)
{
	for (int row = 0; row < model.rowCount(); row++) {
		const QModelIndex index = model.index(row, 0);
		if (index.data(Qt::UserRole).value<PVCol>() == series) {
			return index.data(PVSeriesTreeModel::ValueRole).toString();
		}
	}
	return "no such series";
}

void move_cursor(QWidget* zoomer, int x)
{
	const QPointF position(x, zoomer->height() / 2);
	QMouseEvent move(QEvent::MouseMove, position, zoomer->mapToGlobal(position), Qt::NoButton,
	                 Qt::NoButton, Qt::NoModifier);
	QApplication::sendEvent(zoomer, &move);
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());
	const std::string csv = dir.filePath("power.csv").toStdString();
	const std::string format = dir.filePath("power.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (PVRow row = 0; row < ROW_COUNT; row++) {
			out << row << ',' << (row < ROW_COUNT / 2 ? 7 : row % 2 ? 1 : 9) << '\n';
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();

	// After TestEnv, which runs an application of its own while it builds.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	QApplication app(argc, argv);
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVSeriesViewWidget widget(view, PVCol(0));
	widget.resize(1000, 500);
	widget.show();
	PV_ASSERT_VALID(QTest::qWaitForWindowExposed(&widget));

	// The view samples as many columns as it is wide 200 ms after it was last resized, and the
	// user points at it long after.
	QTest::qWait(500);

	auto* zoomer = widget.findChild<PVParallelView::PVSeriesViewZoomer*>();
	auto* model = widget.findChild<PVSeriesTreeModel*>();
	PV_ASSERT_VALID(zoomer != nullptr and model != nullptr);
	std::cout << "view " << zoomer->width() << " pixels wide" << std::endl;

	// What is shown, once it is the expected value or once the wait is over.
	const auto shows = [&](const QString& expected) {
		(void)QTest::qWaitFor([&] { return value_of(*model, POWER) == expected; }, 5000);
		return value_of(*model, POWER).toStdString();
	};

	move_cursor(zoomer, 0);
	PV_VALID(shows("7"), std::string("7"));

	move_cursor(zoomer, zoomer->width() - 1);
	PV_VALID(shows("1 – 9"), std::string("1 – 9"));

	// The rows out of the selection are not drawn, and their values not shown.
	Squey::PVSelection selection(view->get_row_count());
	selection.select_all();
	for (PVRow row = ROW_COUNT / 2; row < ROW_COUNT; row += 2) {
		selection.set_line(row, false);
	}
	view->set_selection_view(selection);
	// Sampling again for over 250 ms opens a progress box: the cursor leaves the view for it,
	// and the values come back with the next move.
	move_cursor(zoomer, zoomer->width() - 1);
	PV_VALID(shows("1"), std::string("1"));

	// Zoomed in on the first rows, one every 27 columns of pixels or so: the columns within the
	// hole of the crosshair around a row read it, which makes most of them, and the others
	// read nothing.
	zoomer->zoom_in(QRectF(0, 0, 2, zoomer->height()));
	int reading = 0;
	int reading_x = -1;
	for (int x = 0; x < zoomer->width(); x++) {
		move_cursor(zoomer, x);
		const std::string value = value_of(*model, POWER).toStdString();
		PV_ASSERT_VALID(value.empty() or value == "7", "column", x, "value", value);
		if (not value.empty()) {
			reading++;
			reading_x = x;
		}
	}
	std::cout << reading << " columns of " << zoomer->width() << " read a value" << std::endl;
	PV_ASSERT_VALID(reading > zoomer->width() / 2 and reading < zoomer->width(),
	                "columns reading a value", reading);

	move_cursor(zoomer, reading_x);
	PV_VALID(shows("7"), std::string("7"));
	QEvent leave(QEvent::Leave);
	QApplication::sendEvent(zoomer, &leave);
	PV_VALID(shows(""), std::string(""));

	return 0;
}
