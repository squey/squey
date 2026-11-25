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

// Listing only the selected series hides, in the list of a series view, the rows of the
// series left out of the selection, which the plot does not draw. A series selected while
// its row is hidden, as Ctrl+A and the select series mode do, is drawn and listed again.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesTreeWidget.h>
#include <pvparallelview/PVSeriesView.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <squey/PVView.h>

#include <QAction>
#include <QApplication>
#include <QImage>
#include <QTemporaryDir>
#include <QTest>

#include <fstream>
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
   <axis name="steady" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="other" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

const PVCol OTHER(2);

// "other" holds its largest value past the first rows, which draws it along the top of the
// plot, where no other series goes.
constexpr PVRow ROW_COUNT = 10000;

// The list of all the series, rather than the one of the select series mode.
PVSeriesTreeView* list_of(QWidget& widget)
{
	for (auto* list : widget.findChildren<PVSeriesTreeView*>()) {
		if (qobject_cast<PVSeriesTreeModel*>(list->model()) != nullptr) {
			return list;
		}
	}
	return nullptr;
}

std::string listed(const PVSeriesTreeView& list)
{
	std::string names;
	for (int row = 0; row < list.model()->rowCount(); row++) {
		if (not list.isRowHidden(row, QModelIndex())) {
			names += (names.empty() ? "" : " ") +
			         list.model()->index(row, 0).data().toString().toStdString();
		}
	}
	return names;
}

QModelIndex index_of(const QAbstractItemModel& model, PVCol series)
{
	for (int row = 0; row < model.rowCount(); row++) {
		const QModelIndex index = model.index(row, 0);
		if (index.data(Qt::UserRole).value<PVCol>() == series) {
			return index;
		}
	}
	return {};
}

int pixels_of(QWidget& plot, const QColor& color)
{
	const QImage image = plot.grab().toImage().convertToFormat(QImage::Format_RGB32);
	int count = 0;
	for (int y = 0; y < image.height(); y++) {
		const auto* line = reinterpret_cast<const QRgb*>(image.constScanLine(y));
		for (int x = 0; x < image.width(); x++) {
			count += qRgb(qRed(line[x]), qGreen(line[x]), qBlue(line[x])) == color.rgb();
		}
	}
	return count;
}

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());
	const std::string csv = dir.filePath("series.csv").toStdString();
	const std::string format = dir.filePath("series.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (PVRow row = 0; row < ROW_COUNT; row++) {
			out << row << ',' << row % 10 << ',' << (row < ROW_COUNT / 100 ? 0 : 1000) << '\n';
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

	// The view samples as many columns as it is wide 200 ms after it was last resized.
	QTest::qWait(500);

	auto* plot = widget.findChild<PVParallelView::PVSeriesView*>();
	PVSeriesTreeView* list = list_of(widget);
	QAction* selected_only = nullptr;
	for (auto* action : widget.findChildren<QAction*>()) {
		if (action->text() == "Selected series only") {
			selected_only = action;
		}
	}
	PV_ASSERT_VALID(plot != nullptr and list != nullptr and selected_only != nullptr);
	const QModelIndex other = index_of(*list->model(), OTHER);
	const QColor other_color = other.data(Qt::BackgroundRole).value<QBrush>().color();

	// What is listed, once it is the expected series or once the wait is over.
	const auto lists = [&](const std::string& expected) {
		(void)QTest::qWaitFor([&] { return listed(*list) == expected; }, 5000);
		return listed(*list);
	};
	// Whether the plot draws "other", once it is the expected answer or once the wait is over.
	const auto draws_other = [&](bool expected) {
		const auto drawn = [&] { return pixels_of(*plot, other_color) > plot->width() / 2; };
		(void)QTest::qWaitFor([&] { return drawn() == expected; }, 5000);
		return drawn();
	};

	// The abscissa, a number as well, is a series of its own.
	selected_only->trigger();
	PV_VALID(lists("time steady other"), std::string("time steady other"));
	PV_VALID(draws_other(true), true);

	list->selectionModel()->select(other, QItemSelectionModel::Deselect);
	PV_VALID(lists("time steady"), std::string("time steady"));
	PV_VALID(draws_other(false), false);

	list->selectionModel()->select(other, QItemSelectionModel::Select);
	PV_VALID(lists("time steady other"), std::string("time steady other"));
	PV_VALID(draws_other(true), true);

	list->selectionModel()->select(other, QItemSelectionModel::Deselect);
	selected_only->trigger();
	PV_VALID(lists("time steady other"), std::string("time steady other"));
	PV_VALID(draws_other(false), false);

	return 0;
}
