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

// Clicking a column name in the SQL editor offers the other columns. Reading a
// query is how one notices it names the wrong axis, so the click is the
// shortest way from noticing to fixing.
//
// What is pinned here is mostly what the feature must not do: fire on anything
// that is not a column, since a click is also how a cursor is placed, and leave
// half of the old name behind when the new one is picked -- a completion
// replaces what precedes the cursor, so clicking mid-word has to move it first.
//
// Runs under "-platform offscreen" (see CMakeLists.txt).

#include <pvguiqt/PVSqlCodeEditor.h>

#include <pvkernel/core/squey_assert.h>

#include <QAbstractItemView>
#include <QApplication>
#include <QCompleter>
#include <QScrollBar>
#include <QStyle>
#include <QStyleOptionSlider>
#include <QStandardItemModel>
#include <QTest>
#include <QFile>
#include <QTextCursor>
#include <QTextDocument>

#include <string>

#include "test-env.h"

namespace
{

//! The names the popup is currently built from, whatever it filters down to.
QStringList offered(const QStandardItemModel& model)
{
	QStringList names;
	for (int row = 0; row < model.rowCount(); ++row) {
		names << model.item(row)->data(PVGuiQt::PVSqlCodeEditor::InsertRole).toString();
	}
	return names;
}

//! What the popup shows for each of them, which is where the type is described.
QStringList labels(const QStandardItemModel& model)
{
	QStringList shown;
	for (int row = 0; row < model.rowCount(); ++row) {
		shown << model.item(row)->text();
	}
	return shown;
}

//! The category headings the popup currently shows, in order.
QStringList sections(const QStandardItemModel& model)
{
	QStringList titles;
	for (int row = 0; row < model.rowCount(); ++row) {
		if (model.item(row)->data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool()) {
			titles << model.item(row)->data(PVGuiQt::PVSqlCodeEditor::NameRole).toString();
		}
	}
	return titles;
}

//! Accept @a wanted from the popup, the way Enter does.
void pick(QCompleter* completer, const QString& wanted)
{
	QAbstractItemModel* offers = completer->completionModel();
	for (int row = 0; row < offers->rowCount(); ++row) {
		if (offers->index(row, 0).data(PVGuiQt::PVSqlCodeEditor::InsertRole).toString() == wanted) {
			completer->popup()->setCurrentIndex(offers->index(row, 0));
			QTest::keyClick(completer->popup(), Qt::Key_Return);
			QApplication::processEvents();
			return;
		}
	}
	PV_ASSERT_VALID(false, "the popup does not offer", wanted.toStdString());
}

/**
 * Click inside the word covering @a position.
 *
 * The caret rectangle for a position is a thin bar at the character boundary,
 * which is close enough: what matters is that it lands strictly inside the
 * word rather than at either end.
 */
void click_at(PVGuiQt::PVSqlCodeEditor& editor, int position)
{
	QTextCursor cursor = editor.textCursor();
	cursor.setPosition(position);
	const QPoint point = editor.cursorRect(cursor).center();
	QTest::mouseClick(editor.viewport(), Qt::LeftButton, Qt::NoModifier, point);
	QApplication::processEvents();
}

} // namespace

int main(int argc, char** argv)
{
	init_env();
	QApplication app(argc, argv); // argv carries "-platform offscreen"

	PVGuiQt::PVSqlCodeEditor editor;
	// "port" and "port_dst" share a prefix, so picking the second one is a real
	// replacement rather than an append. The dashed name is taken from a pcap
	// format: it cannot be written into a query without quotes.
	editor.set_columns({"rowid", "port", "port_dst", "font-infos-family_name"},
	                   {"BIGINT", "UINTEGER", "UINTEGER", "VARCHAR"});
	editor.resize(600, 200);
	editor.show();
	QApplication::processEvents();

	auto* completer = editor.findChild<QCompleter*>();
	auto* model = editor.findChild<QStandardItemModel*>();
	PV_ASSERT_VALID(completer != nullptr && model != nullptr, "the editor holds a completer", 0);

	// --- A click on a column offers the columns -------------------------------
	{
		editor.setPlainText("port = 80");
		click_at(editor, 2); // inside "port"

		PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open", 0);
		// The cursor sits at the end of the name, which is what lets the whole of
		// it be replaced.
		PV_VALID(editor.textCursor().position(), 4);

		// Columns, and only columns: the click landed on one, so the answer to it
		// is the others -- not the tables, not the keywords.
		const QStringList names = offered(*model);
		PV_ASSERT_VALID(names.contains("port_dst"), "the other columns are missing",
		                names.join(", ").toStdString());
		PV_ASSERT_VALID(not names.contains("layers") && not names.contains("SELECT"),
		                "tables or keywords offered on a column",
		                names.join(", ").toStdString());

		// And every one of them: reaching a different column is what the click is
		// for, so filtering by the name already there would hide the answer.
		PV_ASSERT_VALID(completer->completionPrefix().isEmpty(), "the list is filtered",
		                completer->completionPrefix().toStdString());
		PV_VALID(completer->completionCount(), model->rowCount());
	}

	// --- Picking another one replaces the whole name --------------------------
	// Not the part before the cursor: the click was mid-word, and appending
	// there would leave "port_dstt".
	{
		pick(completer, "port_dst");
		PV_VALID(editor.toPlainText().toStdString(), std::string("port_dst = 80"));
	}

	// --- A name that needs quoting replaces, and is replaced, whole -----------
	// It is written and clicked with its quotes, but it is listed -- and
	// inserted -- under the name the rest of the application shows.
	{
		editor.setPlainText("\"font-infos-family_name\" = 'x'");
		click_at(editor, 10); // inside the quoted name

		PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open on a quoted name",
		                0);
		// Past the closing quote, so the replacement covers the quotes too.
		PV_VALID(editor.textCursor().position(), 24);

		pick(completer, "port");
		PV_VALID(editor.toPlainText().toStdString(), std::string("port = 'x'"));

		// And back the other way: the quotes come from the insertion, not from
		// what was there.
		click_at(editor, 2);
		pick(completer, "font-infos-family_name");
		PV_VALID(editor.toPlainText().toStdString(),
		         std::string("\"font-infos-family_name\" = 'x'"));
	}

	// --- Anything else is just a click ----------------------------------------
	// Each of these starts from an open popup, so what is checked is that the
	// click closed it rather than that it never opened.
	{
		const auto click_away = [&](int on_column, int elsewhere, const char* what) {
			click_at(editor, on_column);
			PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open first", what);
			click_at(editor, elsewhere);
			PV_ASSERT_VALID(not completer->popup()->isVisible(), "a popup where no column is", what);
		};

		editor.setPlainText("port = 80");
		click_away(2, 8, "a literal");
		click_away(2, 5, "an operator");

		editor.setPlainText("SELECT port FROM layers");
		click_away(9, 3, "a keyword");
		click_away(9, 20, "a table name");
	}

	// --- Enter runs, the modified keys break the line -------------------------
	// A query is a line, so that is what Enter should do with it. The line break
	// moves to Ctrl and Shift, which is the bargain every one-line query box
	// makes.
	{
		editor.setPlainText("port = 80");
		int runs = 0;
		QObject::connect(&editor, &PVGuiQt::PVSqlCodeEditor::run_requested, [&runs]() { ++runs; });

		QTextCursor at_end = editor.textCursor();
		at_end.movePosition(QTextCursor::End);
		editor.setTextCursor(at_end);

		QTest::keyClick(&editor, Qt::Key_Return);
		PV_VALID(runs, 1);
		PV_VALID(editor.toPlainText().toStdString(), std::string("port = 80"));

		QTest::keyClick(&editor, Qt::Key_Return, Qt::ShiftModifier);
		PV_VALID(runs, 1);
		PV_VALID(editor.toPlainText().toStdString(), std::string("port = 80\n"));

		QTest::keyClick(&editor, Qt::Key_Return, Qt::ControlModifier);
		PV_VALID(runs, 1);
		PV_VALID(editor.toPlainText().toStdString(), std::string("port = 80\n\n"));

		// The popup owns Enter while it is up: picking a completion must not run
		// the query as a side effect.
		editor.setPlainText("port = 80");
		click_at(editor, 2);
		PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open", 0);
		pick(completer, "port_dst");
		PV_VALID(runs, 1);
		PV_VALID(editor.toPlainText().toStdString(), std::string("port_dst = 80"));
	}

	// --- The layers are offered by name ---------------------------------------
	// A layer is named by a string, so what a query needs is the whole call:
	// the name alone is not something that can be written where a table goes.
	{
		editor.set_layer_provider(
		    []() { return QStringList{"All events", "Night traffic"}; });
		editor.setPlainText("SELECT rowid FROM ");
		QTextCursor at_end = editor.textCursor();
		at_end.movePosition(QTextCursor::End);
		editor.setTextCursor(at_end);
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		QApplication::processEvents();

		const QStringList names = offered(*model);
		PV_ASSERT_VALID(names.contains("layer('All events')"), "a layer is offered whole", 0);
		PV_ASSERT_VALID(names.contains("layer('Night traffic')"), "every layer is offered", 0);
		// The heading is what tells them from the scopes above.
		PV_ASSERT_VALID(sections(*model).contains("Layers"), "the layers are under a heading", 0);

		pick(completer, "layer('Night traffic')");
		PV_VALID(editor.toPlainText().toStdString(),
		         std::string("SELECT rowid FROM layer('Night traffic')"));
	}

	// --- Picking layer() offers the layers rather than choosing one ------------
	// The generic entry is a shape to fill: naming a layer there would pick for
	// the user, and pick wrong as soon as the view holds more than one.
	{
		editor.setPlainText("SELECT rowid FROM ");
		QTextCursor at_end = editor.textCursor();
		at_end.movePosition(QTextCursor::End);
		editor.setTextCursor(at_end);
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		QApplication::processEvents();

		pick(completer, "layer('')");
		PV_VALID(editor.toPlainText().toStdString(),
		         std::string("SELECT rowid FROM layer('')"));
		// Between the quotes, which is where the name goes.
		PV_VALID(editor.textCursor().position(), 25);
		// And the layers are on offer without another keystroke.
		PV_ASSERT_VALID(completer->popup()->isVisible(), "the layers did not open", 0);
		const QStringList names = offered(*model);
		PV_ASSERT_VALID(names.contains("All events") && names.contains("Night traffic"),
		                "the layers are offered", names.join(", ").toStdString());
	}

	// --- And from inside the call ----------------------------------------------
	// What is being typed there is a string, which the word under the cursor
	// does not span: a name holding a space would otherwise be replaced from
	// its last word only.
	{
		editor.setPlainText("SELECT rowid FROM layer('Night tr");
		QTextCursor at_end = editor.textCursor();
		at_end.movePosition(QTextCursor::End);
		editor.setTextCursor(at_end);
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		QApplication::processEvents();

		// Matched on the whole literal, so the name with the space is the one
		// left standing.
		const QStringList names = offered(*model);
		PV_ASSERT_VALID(names.contains("Night traffic"), "the name is offered bare in the call",
		                0);
		PV_ASSERT_VALID(not names.contains("All events"), "filtered", "on the whole literal");

		pick(completer, "Night traffic");
		PV_VALID(editor.toPlainText().toStdString(),
		         std::string("SELECT rowid FROM layer('Night traffic')"));
	}

	// --- The case that was typed says which of two it is -----------------------
	// "selection" and SELECT share a beginning, and both stay on offer: what
	// tells them apart is how they were spelt, so that is what the selection
	// follows.
	{
		const auto typed_selection = [&](const QString& text) {
			editor.setPlainText(text);
			QTextCursor at_end = editor.textCursor();
			at_end.movePosition(QTextCursor::End);
			editor.setTextCursor(at_end);
			QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
			QApplication::processEvents();
			return completer->popup()
			    ->currentIndex()
			    .data(PVGuiQt::PVSqlCodeEditor::InsertRole)
			    .toString();
		};

		PV_VALID(typed_selection("sele").toStdString(), std::string("selection"));
		PV_VALID(typed_selection("SELE").toStdString(), std::string("SELECT"));
		// Both are there either way: the case picks, it does not filter.
		PV_ASSERT_VALID(offered(*model).contains("selection"),
		                "an upper-case prefix still offers the scope", 0);
	}

	// --- Categories are headings, not entries ----------------------------------
	{
		editor.setPlainText("");
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		QApplication::processEvents();

		const QStringList titles = sections(*model);
		PV_ASSERT_VALID(titles.contains("Clauses") && titles.contains("Operators"),
		                "the keywords are grouped", titles.join(", ").toStdString());
		// A heading is not something one can pick: it carries nothing to insert
		// and the arrow keys have to step over it.
		for (int row = 0; row < model->rowCount(); ++row) {
			if (not model->item(row)->data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool()) {
				continue;
			}
			PV_ASSERT_VALID(
			    model->item(row)->data(PVGuiQt::PVSqlCodeEditor::InsertRole).toString().isEmpty(),
			    "a heading inserts nothing", row);
			PV_ASSERT_VALID(not model->item(row)->isSelectable(), "a heading is not selectable",
			                row);
		}
		// And the first thing selected is a real entry rather than a heading.
		const QModelIndex current = completer->popup()->currentIndex();
		PV_ASSERT_VALID(current.isValid(), "something is selected", 0);
		PV_ASSERT_VALID(not current.data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool(),
		                "a heading is not what gets selected", current.row());
	}

	// --- A business type says what it is, and how to reach it -----------------
	// An address is exposed as the integer it is stored as, so "UINTEGER" alone
	// would leave it indistinguishable from a counter -- and would not say that a
	// conversion exists.
	{
		editor.set_columns({"rowid", "src", "count"}, {"BIGINT", "UINTEGER", "UINTEGER"},
		                   {"", "ipv4", "number_uint32"});
		editor.setPlainText("src = 0");
		click_at(editor, 1);
		PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open", 0);

		const QStringList shown = labels(*model);
		PV_ASSERT_VALID(shown.filter("src — IPv4").size() == 1, "the address is not named as one",
		                shown.join(" | ").toStdString());
		PV_ASSERT_VALID(shown.filter("count — UINTEGER").size() == 1,
		                "a plain number gained a description it does not have",
		                shown.join(" | ").toStdString());

		// Both directions offered, the literal-side one first: it is the one that
		// leaves the comparison on the stored integer.
		const QStringList names = offered(*model);
		PV_ASSERT_VALID(names.indexOf("ipv4()") >= 0 && names.indexOf("ipv4_text()") >= 0,
		                "the conversions are not offered", names.join(", ").toStdString());
		PV_ASSERT_VALID(names.indexOf("ipv4()") < names.indexOf("ipv4_text()"),
		                "the column-side conversion comes first",
		                names.join(", ").toStdString());
		// Nothing for a type that has none.
		PV_ASSERT_VALID(names.filter("mac_address").isEmpty(),
		                "a conversion offered for a type no column carries",
		                names.join(", ").toStdString());

		// Picking one leaves the cursor where the argument goes.
		pick(completer, "ipv4()");
		PV_VALID(editor.toPlainText().toStdString(), std::string("ipv4() = 0"));
		PV_VALID(editor.textCursor().position(), 5);
	}

	// --- The popup and the help page say the same sentence ---------------------
	// Aligned by hand in three places -- this header's own documentation, the
	// table above, and the help resources -- because they address one reader.
	// Nothing but this holds them together, so the day one is reworded the other
	// stops recognising it, and a user has two phrasings to map onto each other.
	{
		QString help;
		for (const char* page : {":help-sql-console-tables", ":help-sql-console-types"}) {
			QFile file(page);
			PV_ASSERT_VALID(file.open(QIODevice::ReadOnly), "help page missing", page);
			help += QString::fromUtf8(file.readAll());
		}
		// Rendered rather than raw: the page wraps its cells, so a sentence is
		// split across lines by whitespace the reader never sees.
		QTextDocument rendered;
		rendered.setHtml(help);
		const QString shown = rendered.toPlainText().simplified();

		editor.setPlainText("SELECT * FROM ");
		QTextCursor at_end = editor.textCursor();
		at_end.movePosition(QTextCursor::End);
		editor.setTextCursor(at_end);
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		PV_ASSERT_VALID(completer->popup()->isVisible(), "the table list did not open", 0);

		int checked = 0;
		for (int row = 0; row < model->rowCount(); ++row) {
			if (model->item(row)->data(PVGuiQt::PVSqlCodeEditor::SectionRole).toBool()) {
				continue;
			}
			const QString detail =
			    model->item(row)->data(PVGuiQt::PVSqlCodeEditor::DetailRole).toString();
			if (detail.isEmpty()) {
				continue;
			}
			PV_ASSERT_VALID(shown.contains(detail), "the help page words it otherwise",
			                detail.toStdString());
			++checked;
		}
		PV_ASSERT_VALID(checked > 0, "no table was offered to check", checked);
	}

	// --- The scroll bar stays inside the rounded background -------------------
	// The background is painted on the viewport. A scroll area lays its bar out
	// in a strip taken from the widget's edge, which the viewport is not part of,
	// so a bar laid out that way was drawn on the transparent pixels outside the
	// corners. It floats over the list instead, and this is what says so.
	{
		// More entries than fit, so a bar has something to scroll.
		editor.setPlainText("");
		QTest::keyClick(&editor, Qt::Key_Space, Qt::ControlModifier);
		QApplication::processEvents();
		PV_ASSERT_VALID(completer->popup()->isVisible(), "the popup did not open", 0);

		QScrollBar* bar = nullptr;
		for (QScrollBar* candidate : completer->popup()->findChildren<QScrollBar*>()) {
			if (candidate->isVisible() && candidate->orientation() == Qt::Vertical) {
				bar = candidate;
			}
		}
		PV_ASSERT_VALID(bar != nullptr, "no scroll bar to check -- the list fits",
		                completer->popup()->model()->rowCount());

		// --- The handle is a pill, against the theme's own rule -------------
		// The theme styles scroll bar handles too, and ours only wins over it
		// because it is set on the bar itself: the same rule set through the
		// popup's sheet lost to the application's, whatever the documented
		// precedence says, and the handle came out square and inset by a pixel.
		// So the theme is loaded here, or the test would pass against a cascade
		// the application never has.
		{
			QFile theme(":/theme-dark.qss");
			PV_ASSERT_VALID(theme.open(QIODevice::ReadOnly), "the theme is not in the resources", 0);
			qApp->setStyleSheet(QString::fromUtf8(theme.readAll()));
			QApplication::processEvents();

			QStyleOptionSlider opt;
			opt.initFrom(bar);
			opt.orientation = Qt::Vertical;
			opt.minimum = bar->minimum();
			opt.maximum = bar->maximum();
			opt.sliderPosition = bar->value();
			opt.sliderValue = bar->value();
			opt.pageStep = bar->pageStep();
			const QRect handle = bar->style()->subControlRect(QStyle::CC_ScrollBar, &opt,
			                                                  QStyle::SC_ScrollBarSlider, bar);
			// Full width: the theme's rule inset it by a pixel when it won.
			PV_VALID(handle.width(), bar->width());

			const QImage drawn = bar->grab().toImage();
			// A corner the radius cut away, and the middle it left alone.
			PV_VALID(int(drawn.pixelColor(handle.x(), handle.y()).alpha()), 0);
			PV_VALID(int(drawn.pixelColor(handle.center().x(), handle.center().y()).alpha()), 255);
		}

		const QRect painted = completer->popup()->viewport()->geometry();
		PV_ASSERT_VALID(painted.contains(bar->geometry()),
		                "the bar reaches outside the painted background",
		                bar->geometry().right(), "background", painted.right());
		// And clear of the corners, where the radius cuts the background away.
		PV_ASSERT_VALID(bar->geometry().top() > painted.top() &&
		                    bar->geometry().bottom() < painted.bottom(),
		                "the bar runs into the rounded corners", bar->geometry().top(),
		                "background", painted.top());
	}

	return 0;
}
