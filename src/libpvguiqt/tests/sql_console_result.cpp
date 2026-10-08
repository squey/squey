/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// A GROUP BY typed in the SQL console is shown through the very dialog the
// distinct-values listing uses, and in the same place: a dock of the workspace,
// beside the views the query was written to explain.
//
// Sharing that dialog also brings along the part of it that reads a listed
// value back into a column of the source to select rows. That part cannot work
// here: a query groups over an expression of its own choosing, not over an
// axis, so there is no column to search in and the rows it would select bear no
// relation to what is displayed.
//
// This test pins that, and where things land: the console at the bottom across
// the whole width, its result docked beside the views rather than floating over
// them.
//
// Runs under "-platform offscreen" (see CMakeLists.txt).

#include <squey/PVRoot.h>
#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVTheme.h>
#include <pvkernel/core/squey_assert.h>

#include <pvbase/types.h>

#include <pvdisplays/PVDisplayIf.h>

#include <pvguiqt/PVAbstractListStatsDlg.h>
#include <pvguiqt/PVAbstractTableView.h>
#include <pvguiqt/PVDisplayViewDistinctValues.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>
#include <pvguiqt/PVDisplayViewSqlConsole.h>
#include <pvguiqt/PVLayerFilterProcessWidget.h>
#include <pvguiqt/PVSqlCodeEditor.h>
#include <pvguiqt/PVStatsModel.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/common.h>

#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/widgets/PVHelpWidget.h>

#include <QAbstractItemDelegate>
#include <QAction>
#include <QApplication>
#include <QGroupBox>
#include <QKeyEvent>
#include <QTextBrowser>
#include <QMenu>
#include <QFontMetrics>
#include <QPushButton>
#include <QToolButton>
#include <QTextDocument>
#include <QSizePolicy>

#include <any>
#include <string>
#include <vector>

#include "common.h"
#include "test-env.h"

namespace
{

//! Every action reachable from the dialog's menus, whichever menu holds it.
QStringList menu_actions(const PVGuiQt::PVAbstractListStatsDlg& dlg)
{
	QStringList names;
	for (const QMenu* menu : dlg.findChildren<QMenu*>()) {
		for (const QAction* act : menu->actions()) {
			names << act->text();
		}
	}
	return names;
}

} // namespace

int main(int argc, char** argv)
{
	init_env();

	Squey::PVRoot root;
	const QString file = QString(TEST_FOLDER) + "/picviz/heat_line.csv";
	Squey::PVSource& src = get_src_from_file(root, file, file + ".format");
	Squey::PVView& view = src.emplace_add_child().emplace_add_child().emplace_add_child();

	QApplication app(argc, argv); // argv carries "-platform offscreen"
	PVGuiQt::common::register_displays();

	// The selection the view starts with, kept to tell whether anything the
	// dialog does reaches it.
	const Squey::PVSelection initial_selection = view.get_real_output_selection();

	// --- The SQL console: a group-by result ----------------------------------
	// Opened the way the toolbar opens it, so the console sits in a dock of the
	// workspace: that is what lets its result be docked next to it rather than
	// float over the window.
	PVGuiQt::PVWorkspaceBase workspace(nullptr);
	workspace.resize(1200, 800);
	// A central display and a side dock, the way a source workspace opens: with
	// neither, a dock is alone in the window and every claim about where it sits
	// is a claim about nothing. A bare widget stands in for the listing -- what is
	// checked here is the layout, not what fills it.
	{
		auto* central = new QWidget;
		central->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
		workspace.set_central_display(&view, central, false, true);
	}
	workspace.create_view_widget(PVDisplays::display_view_if<PVDisplays::PVDisplayViewLayerStack>(),
	                             &view);
	workspace.create_view_widget(PVDisplays::display_view_if<PVDisplays::PVDisplayViewSqlConsole>(),
	                             &view);
	workspace.show();
	QApplication::processEvents();

	auto* editor = workspace.findChild<PVGuiQt::PVSqlCodeEditor*>();
	PV_ASSERT_VALID(editor != nullptr, "editor", 0);

	// --- The console sits at the bottom, across everything --------------------
	// A query is written about what is above it, and it is a line of text rather
	// than a picture: wide and short is the shape it wants.
	{
		auto* console_dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(editor);
		PV_ASSERT_VALID(console_dock != nullptr, "the console is not docked", 0);
		PV_VALID(int(workspace.dockWidgetArea(console_dock)), int(Qt::BottomDockWidgetArea));

		// The whole width: Qt gives the bottom corners to the bottom area, so
		// this holds as long as nobody hands them to the sides.
		PV_ASSERT_VALID(console_dock->width() == workspace.width(),
		                "the console does not run the whole width", console_dock->width(),
		                "workspace", workspace.width());

		// And one line of text tall: the bottom area otherwise takes whatever the
		// layout leaves it, which on a fresh workspace was two thirds of the
		// height. The dock can be pulled up, so this is a starting size, not a
		// ceiling.
		const int line = QFontMetrics(editor->document()->defaultFont()).lineSpacing();
		PV_ASSERT_VALID(editor->height() < 2 * line, "the console did not open on one line",
		                editor->height(), "line", line);
		PV_ASSERT_VALID(console_dock->height() < workspace.height() / 4,
		                "the console took over the window", console_dock->height(), "workspace",
		                workspace.height());

		// And the side dock stops above it rather than beside it.
		PVGuiQt::PVViewDisplay* side = nullptr;
		for (auto* dock : workspace.findChildren<PVGuiQt::PVViewDisplay*>()) {
			if (workspace.dockWidgetArea(dock) == Qt::RightDockWidgetArea) {
				side = dock;
			}
		}
		PV_ASSERT_VALID(side != nullptr, "no side dock to compare against", 0);
		PV_ASSERT_VALID(side->geometry().bottom() <= console_dock->geometry().top(),
		                "the side dock reaches past the console", side->geometry().bottom(),
		                "console top", console_dock->geometry().top());
	}
	const QString sql =
	    "SELECT uint8 AS v, COUNT(*) AS n FROM layers GROUP BY 1 ORDER BY n DESC LIMIT 20";
	editor->setPlainText(sql);

	// The console's only button, and it carries an icon rather than a label.
	auto* run = editor->parentWidget()->findChild<QPushButton*>();
	PV_ASSERT_VALID(run != nullptr, "run button", 0);
	run->click();
	QApplication::processEvents();

	// A value/count result goes to the stats dialog rather than to the plain
	// grid, which is what brings the count column's delegate along.
	auto* sql_dlg = workspace.findChild<PVGuiQt::PVAbstractListStatsDlg*>();
	PV_ASSERT_VALID(sql_dlg != nullptr, "sql result dialog", 0);

	// It has to be inside a dock of that workspace, and titled by the query --
	// results stack as tabs, so a constant title would leave them apart only by
	// position.
	auto* dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(sql_dlg);
	PV_ASSERT_VALID(dock != nullptr, "result not docked", 0);
	PV_ASSERT_VALID(sql.startsWith(dock->windowTitle().chopped(1)), "dock title",
	                dock->windowTitle().toStdString());
	PV_ASSERT_VALID(dock->windowTitle().size() <= 40, "dock title too long for a tab",
	                dock->windowTitle().size());
	PV_ASSERT_VALID(not sql_dlg->isWindow(), "result is a window of its own", 1);

	// The console kept its own dock: the result was added beside it, so four in
	// all with the central display and the layer stack.
	PV_ASSERT_VALID(workspace.findChildren<PVGuiQt::PVViewDisplay*>().size() == 4, "docks",
	                workspace.findChildren<PVGuiQt::PVViewDisplay*>().size());

	// The count column is painted, not written as text: without its delegate it
	// renders blank however well the query ran.
	PV_ASSERT_VALID(sql_dlg->_values_view->itemDelegateForColumn(1) != nullptr, "count delegate",
	                0);
	PV_ASSERT_VALID(sql_dlg->model().size() > 0, "rows", sql_dlg->model().size());

	// None of the three routes from a listed value to a view selection may be
	// offered: the two menu entries, and the range picker in the "Selection"
	// group box.
	const QStringList sql_actions = menu_actions(*sql_dlg);
	PV_ASSERT_VALID(not sql_actions.contains("Search for this value"), "search offered",
	                sql_actions.join(", ").toStdString());
	PV_ASSERT_VALID(not sql_actions.contains("Create one layer with those values"),
	                "layer creation offered", sql_actions.join(", ").toStdString());
	PV_ASSERT_VALID(not sql_dlg->_select_groupbox->isVisibleTo(sql_dlg), "selection group box", 1);

	// Copying is what the listing's row selection is still for, so the entry
	// that drives it has to survive.
	PV_ASSERT_VALID(sql_actions.contains("Copy values"), "copy offered",
	                sql_actions.join(", ").toStdString());

	// Committing a row selection must therefore stop at the dialog: no layer
	// filter is run, and the view keeps the selection it had.
	sql_dlg->model().current_selection().select_all();
	Q_EMIT sql_dlg->_values_view->selection_commited();
	QApplication::processEvents();

	PV_ASSERT_VALID(sql_dlg->findChild<PVGuiQt::PVLayerFilterProcessWidget*>() == nullptr,
	                "layer filter run", 1);
	// A selection is a bit field with no equality operator, so the difference is
	// spelled out: XOR leaves a bit wherever the two disagree.
	PV_ASSERT_VALID((view.get_real_output_selection() ^ initial_selection).is_empty(),
	                "view selection changed", 1);

	// --- The distinct-values listing: everything stays ------------------------
	// Same dialog, same code: what was switched off above has to be untouched
	// where a listed value really is a value of a column.
	QWidget* distinct = PVDisplays::get_widget(
	    PVDisplays::display_view_if<PVDisplays::PVDisplayViewDistinctValues>(), &view,
	    static_cast<QWidget*>(nullptr), PVDisplays::PVDisplayViewIf::Params{std::any(PVCombCol(0))});
	auto* distinct_dlg = qobject_cast<PVGuiQt::PVAbstractListStatsDlg*>(distinct);
	PV_ASSERT_VALID(distinct_dlg != nullptr, "distinct values dialog", 0);

	const QStringList distinct_actions = menu_actions(*distinct_dlg);
	PV_ASSERT_VALID(distinct_actions.contains("Search for this value"), "search missing",
	                distinct_actions.join(", ").toStdString());
	PV_ASSERT_VALID(distinct_actions.contains("Create one layer with those values"),
	                "layer creation missing", distinct_actions.join(", ").toStdString());
	distinct_dlg->show();
	QApplication::processEvents();
	PV_ASSERT_VALID(distinct_dlg->_select_groupbox->isVisibleTo(distinct_dlg),
	                "selection group box missing", 0);


	// --- The dock carries a help page -----------------------------------------
	// Which is the "?" in its title bar, and which posts a help key to the widget
	// -- the same route every other view's help takes. The console is one line
	// tall, so the dock opens for as long as the page is up and closes back
	// afterwards: an overlay of its usual size would be a sliver.
	{
		auto* console_dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(editor);
		PV_ASSERT_VALID(console_dock->has_help_page(), "no help button in the dock title", 0);

		QWidget* console = editor->parentWidget();
		const int closed_height = console_dock->height();

		QKeyEvent help(QEvent::KeyPress, Qt::Key_Help, Qt::NoModifier, "?");
		QApplication::sendEvent(console, &help);
		QApplication::processEvents();

		// dynamic_cast rather than findChild: PVHelpWidget carries no Q_OBJECT.
		PVWidgets::PVHelpWidget* page = nullptr;
		for (QWidget* child : console->findChildren<QWidget*>()) {
			if (auto* candidate = dynamic_cast<PVWidgets::PVHelpWidget*>(child)) {
				page = candidate;
			}
		}
		PV_ASSERT_VALID(page != nullptr, "the help page was never built", 0);
		PV_ASSERT_VALID(page->isVisible(), "the help page did not open", 0);

		// The dock made room, and the page covers the console rather than the
		// window: it is the console's page.
		PV_ASSERT_VALID(console_dock->height() > closed_height, "the dock did not open",
		                console_dock->height(), "closed", closed_height);
		PV_ASSERT_VALID(page->parentWidget() == console, "the page is not the console's", 0);

		// And it says something: a missing resource alias opens an empty panel,
		// which looks like a working button.
		auto* rendered = page->findChild<QTextBrowser*>();
		PV_ASSERT_VALID(rendered != nullptr, "the help page has no text view", 0);
		const QString text = rendered->toPlainText();
		for (const char* mentioned : {"layer('All events')", "Ctrl+Enter", "ipv4", "rowid",
		                              "HAVING COUNT(DISTINCT", "to_timestamp"}) {
			PV_ASSERT_VALID(text.contains(mentioned), "missing from the help page", mentioned);
		}

		// The worked example has to be valid SQL. Its columns are invented, so the
		// engine should object to a column and not to the syntax -- a page that
		// teaches a query which does not parse teaches the wrong thing.
		{
			const QString head = "SELECT rowid FROM layers";
			const QString tail = "= 1)";
			const int begin = text.indexOf(head);
			const int end = text.indexOf(tail, begin);
			PV_ASSERT_VALID(begin >= 0 && end > begin, "the example is not on the page", 0);
			const std::string example =
			    text.mid(begin, end - begin + tail.size()).toStdString();

			std::string complaint;
			try {
				Squey::PVDuckDBQuery(view).run_tabular(example, nullptr, 1);
			} catch (const std::exception& e) {
				complaint = e.what();
			}
			PV_ASSERT_VALID(not complaint.empty(), "the example named a real column", example);
			PV_ASSERT_VALID(complaint.find("syntax error") == std::string::npos &&
			                    complaint.find("Parser Error") == std::string::npos,
			                "the example does not parse", complaint);
		}

		// Closing gives the height back.
		page->hide();
		QApplication::processEvents();
		PV_VALID(console_dock->height(), closed_height);
	}

	// --- The console is one per view, shown or not ----------------------------
	// Its toolbar button is a state: pressing it again puts the dock away rather
	// than stacking a second, empty console beside the first.
	{
		auto& display = PVDisplays::display_view_if<PVDisplays::PVDisplayViewSqlConsole>();
		auto* console_dock = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(editor);
		const auto dock_count = workspace.findChildren<PVGuiQt::PVViewDisplay*>().size();

		QToolButton button;
		button.setCheckable(true);
		button.setChecked(true);

		workspace.toggle_unique_view_widget(&button, display, &view);
		QApplication::processEvents();
		PV_ASSERT_VALID(not console_dock->isVisible(), "the console stayed up", 1);
		PV_ASSERT_VALID(not button.isChecked(), "the button stayed pressed", 1);
		PV_VALID(workspace.findChildren<PVGuiQt::PVViewDisplay*>().size(), dock_count);

		workspace.toggle_unique_view_widget(&button, display, &view);
		QApplication::processEvents();
		PV_ASSERT_VALID(console_dock->isVisible(), "the console did not come back", 0);
		PV_ASSERT_VALID(button.isChecked(), "the button did not come back", 0);
		PV_VALID(workspace.findChildren<PVGuiQt::PVViewDisplay*>().size(), dock_count);

		// The same widget, so the query that was typed is still there.
		PV_VALID(editor->toPlainText().toStdString(), sql.toStdString());
	}

	// --- The button carries an icon -------------------------------------------
	// A name with nothing behind it in the resources is not an error: the icon
	// comes out empty, and a toolbar button with nothing drawn on it looks like
	// a gap rather than a mistake. Both themes, since each has its own file.
	{
		const QIcon icon =
		    PVDisplays::display_view_if<PVDisplays::PVDisplayViewSqlConsole>().toolbar_icon();
		const PVCore::PVTheme::EColorScheme was = PVCore::PVTheme::color_scheme();
		for (PVCore::PVTheme::EColorScheme scheme :
		     {PVCore::PVTheme::EColorScheme::LIGHT, PVCore::PVTheme::EColorScheme::DARK}) {
			PVCore::PVTheme::set_color_scheme(scheme);
			const QImage drawn = icon.pixmap(QSize(32, 32)).toImage();
			PV_ASSERT_VALID(not drawn.isNull(), "the toolbar icon resolves to nothing",
			                int(scheme));
			// Drawn rather than merely there: a name with no file behind it
			// gives a pixmap of the right size with nothing on it.
			int opaque = 0;
			for (int y = 0; y < drawn.height(); ++y) {
				for (int x = 0; x < drawn.width(); ++x) {
					opaque += static_cast<int>(qAlpha(drawn.pixel(x, y)) > 200);
				}
			}
			PV_ASSERT_VALID(opaque > 0, "nothing was drawn on the toolbar icon", int(scheme));
		}
		PVCore::PVTheme::set_color_scheme(was);
	}

	return 0;
}
