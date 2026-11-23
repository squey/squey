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
//

// The plot of a series view draws the series it was last given, even when it was painted
// after they were sampled and before it was given them.
//
// Changing the series of the list samples them again, then gives them to the plot. A
// sampling of over 250 ms opens a progress box, whose event loop painted the plot with the
// series of before: the plot kept that picture, the series taken out drawn and those added
// not, until a sampling drew it again.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesTreeWidget.h>
#include <pvparallelview/PVSeriesView.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QImage>
#include <QTest>

#include <iostream>

#include "common.h"

namespace
{

// Whether the plot draws nothing but its background.
bool blank(const QImage& picture)
{
	for (int y = 0; y < picture.height(); y++) {
		for (int x = 0; x < picture.width(); x++) {
			if (picture.pixel(x, y) != picture.pixel(0, 0)) {
				return false;
			}
		}
	}
	return true;
}

} // namespace

int main(int argc, char** argv)
{
	pvtest::TestEnv env(TEST_FOLDER "/picviz/timeserie_fusion.csv",
	                    TEST_FOLDER "/picviz/timeserie_fusion.csv.format", 1,
	                    pvtest::ProcessUntil::View);
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

	auto* plot = widget.findChild<PVParallelView::PVSeriesView*>();
	auto* model = widget.findChild<PVSeriesTreeModel*>();
	PV_ASSERT_VALID(plot != nullptr and model != nullptr);
	const auto picture = [&] { return plot->grab().toImage(); };

	// Every series, once sampled.
	PV_ASSERT_VALID(QTest::qWaitFor([&] { return not blank(picture()); }, 5000));

	plot->show_series({});
	PV_ASSERT_VALID(blank(picture()), "the plot", "draws the series it no longer shows");

	const PVCol series = model->index(0, 0).data(Qt::UserRole).value<PVCol>();
	plot->show_series({{size_t(series), QColor(Qt::red)}});
	PV_ASSERT_VALID(not blank(picture()), "the plot", "does not draw the series it shows");

	std::cout << "the plot draws the series it shows, and only them" << std::endl;
	return 0;
}
