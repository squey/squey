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

// Naming a set of series in the tree of a series view, some of them already
// selected and some not, selects the whole set.
//
// Selecting a series selects its parts with it, and the tree did so by making
// what had just been selected, parts included, Qt's current selection. After a
// ClearAndSelect the current selection is the whole set named, so the series kept
// from before were dropped and only the added ones stayed selected.

#include <pvkernel/core/squey_assert.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesTreeWidget.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <QApplication>
#include <QItemSelectionModel>

#include <iostream>

#include "common.h"

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

	// The abscissa selector asks the display registry whether the series view takes
	// a given axis, and the registry is filled when the backend comes up.
	PVParallelView::common::RAII_backend_init backend_resources;

	PVParallelView::PVSeriesViewWidget widget(view, PVCol(0));
	auto* tree = widget.findChild<PVSeriesTreeView*>();
	PV_ASSERT_VALID(tree != nullptr, "the series tree", "was not found");
	QAbstractItemModel& model = *tree->model();
	QItemSelectionModel& selection = *tree->selectionModel();
	PV_ASSERT_VALID(model.rowCount() > 1, "series offered", model.rowCount());

	const QModelIndex kept = model.index(0, 0);
	const QModelIndex added = model.index(1, 0);
	selection.select(kept, QItemSelectionModel::ClearAndSelect);
	PV_ASSERT_VALID(selection.isSelected(kept), "the first series", "was not selected");

	QItemSelection both(kept, kept);
	both.select(added, added);
	selection.select(both, QItemSelectionModel::ClearAndSelect);
	PV_ASSERT_VALID(selection.isSelected(kept), "the series kept", "was dropped");
	PV_ASSERT_VALID(selection.isSelected(added), "the series added", "was not selected");

	std::cout << "a set of series named, kept and added alike, stays selected" << std::endl;
	return 0;
}
