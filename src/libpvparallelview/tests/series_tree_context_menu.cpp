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

// A right click on the series list of a series view, below the last series.
//
// The list opens a menu of the views a series can be shown in, and takes the
// column of the series from the item under the pointer. Below the last series
// there is no item, hence no column: the menu would be built for the invalid one,
// and the entry of the series view, the first to read the type of the axis it is
// offered for, reads it far past the end of the axes.
//
// No menu opens there. On a series, the menu still offers its views.

#include <pvkernel/core/squey_assert.h>

#include <pvdisplays/PVDisplaysContainer.h>

#include <pvparallelview/PVDisplayViewTimeseries.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesTreeWidget.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <squey/PVView.h>

#include <QAction>
#include <QApplication>
#include <QContextMenuEvent>
#include <QMenu>
#include <QTemporaryDir>

#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <string>

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
   <axis name="falling" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr PVRow ROWS = 100;

/**
 * What a series view looks for among its ancestors to open the views of a series.
 */
class Workspace : public PVDisplays::PVDisplaysContainer
{
  public:
	Workspace() : PVDisplays::PVDisplaysContainer(nullptr) {}

	void
	create_view_widget(PVDisplays::PVDisplayViewIf&, Squey::PVView*, std::vector<std::any>) override
	{
	}
};

/**
 * The entries of the menu a right click at pos opens on the tree, or none if it
 * opens no menu.
 *
 * A menu runs an event loop of its own until it closes: the call queued before the
 * click reads the menu and closes it from inside that loop.
 */
std::optional<QStringList> context_menu(PVSeriesTreeView& tree, QPoint pos)
{
	std::optional<QStringList> entries;
	bool clicking = true;
	std::function<void()> read_and_close;
	read_and_close = [&]() {
		if (auto* menu = qobject_cast<QMenu*>(QApplication::activePopupWidget())) {
			entries.emplace();
			for (QAction* action : menu->actions()) {
				entries->append(action->text());
			}
			menu->close();
		} else if (clicking) {
			QMetaObject::invokeMethod(qApp, read_and_close, Qt::QueuedConnection);
		}
	};
	QMetaObject::invokeMethod(qApp, read_and_close, Qt::QueuedConnection);

	QContextMenuEvent click(QContextMenuEvent::Mouse, pos, tree.viewport()->mapToGlobal(pos));
	QApplication::sendEvent(tree.viewport(), &click);
	clicking = false;
	QCoreApplication::sendPostedEvents(qApp, QEvent::MetaCall);

	return entries;
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
			out << 1000000000 + i * 60 << "," << i << "," << ROWS - i << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();

	// After TestEnv, which runs an application of its own while it builds. On the
	// processor, which every machine has.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	if (qEnvironmentVariableIsEmpty("FORCE_CPU")) {
		qputenv("FORCE_CPU", "1");
	}
	QApplication app(argc, argv);

	// The menu of a series offers the views registered when the backend comes up.
	PVParallelView::common::RAII_backend_init backend_resources;

	Workspace workspace;
	workspace.setCentralWidget(new PVParallelView::PVSeriesViewWidget(view, PVCol(0)));
	workspace.resize(800, 600);
	workspace.show();
	QApplication::processEvents();

	PVSeriesTreeView* tree = nullptr;
	for (PVSeriesTreeView* candidate : workspace.findChildren<PVSeriesTreeView*>()) {
		if (candidate->contextMenuPolicy() == Qt::CustomContextMenu) {
			tree = candidate;
		}
	}
	PV_ASSERT_VALID(tree != nullptr, "the series list", "was not found");
	const QAbstractItemModel& model = *tree->model();
	PV_ASSERT_VALID(model.rowCount() == 2, "series listed", model.rowCount());

	const QRect last = tree->visualRect(model.index(model.rowCount() - 1, 0));
	const QPoint below(last.center().x(), last.bottom() + last.height());
	PV_ASSERT_VALID(tree->viewport()->rect().contains(below), "the series list",
	                "has no room below its last series");
	PV_ASSERT_VALID(not tree->indexAt(below).isValid(), "a series", "lies below the last one");

	const QString series_view =
	    PVDisplays::display_view_if<PVDisplays::PVDisplayViewTimeseries>().axis_menu_name();
	const std::optional<QStringList> on_series = context_menu(*tree, last.center());
	PV_ASSERT_VALID(on_series.has_value(), "a right click on a series", "opens no menu");
	PV_ASSERT_VALID(on_series->contains(series_view), "the menu of a series", "lacks", "the entry",
	                series_view.toStdString());

	const std::optional<QStringList> below_series = context_menu(*tree, below);
	PV_ASSERT_VALID(not below_series.has_value(), "a right click below the last series",
	                "opens a menu");

	std::cout << "a right click on the series list opens a menu on a series only" << std::endl;
	return 0;
}
