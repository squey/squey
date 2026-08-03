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

// A query names the rows it reads the way the application names them: what is
// selected, what the layer stack lets through, and a layer by its name. What is
// checked here is that those names keep meaning that -- they are resolved when
// the query runs, so the same query object answers differently once the
// selection moves or a layer is hidden.
//
// And that the text form of a scope shows a cell the format could not read.
// Such a cell has no value to expose: the typed form gives NULL, because what
// the storage holds for it is an encoding rather than a number. pvcop kept the
// text it was written with, which is what the listing shows and what
// "text := true" hands to SQL.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVLayer.h>
#include <squey/PVLayerStack.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <string>
#include <vector>

#include "common.h"

// Thirteen rows over two axes holding the same text: six readable numbers, four
// words and three empty cells. col1 is typed as a number, so its unreadable
// cells have no numeric form; col2 is text, where every cell reads back as
// written.
const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

constexpr PVCol NUMBER_COL(0);

static size_t count_rows(const Squey::PVDuckDBQuery& query, const std::string& from)
{
	const auto table = query.run_tabular("SELECT COUNT(*) FROM " + from);
	return std::stoull(table.rows[0][0]);
}

/**
 * Show or hide the layer named @a name, and recompute what the stack lets
 * through.
 */
static void set_visible(Squey::PVView* view, const QString& name, bool visible)
{
	const Squey::PVLayerStack& stack = view->get_layer_stack();
	for (int i = 0; i < stack.get_layer_count(); ++i) {
		if (stack.get_layer_n(i).get_name() != name) {
			continue;
		}
		if ((view->get_layer_stack_layer_n_visible_state(i) != 0) != visible) {
			view->toggle_layer_stack_layer_n_visible_state(i);
		}
		view->process_layer_stack();
		return;
	}
	PV_ASSERT_VALID(false, "the test names a layer that exists", name.toStdString());
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	// Bound to the view, which is what carries a selection and a layer stack.
	Squey::PVDuckDBQuery query(*view);
	const std::string num = Squey::PVDuckDBQuery::quote_identifier(query.column_names()[1]);
	const std::string txt = Squey::PVDuckDBQuery::quote_identifier(query.column_names()[2]);

	// --- The base layer is every row ------------------------------------------
	// It is locked, so "layer('All events')" is the way to read the whole source
	// whatever else the stack holds.
	PV_VALID(count_rows(query, "layer('All events')"), row_count);
	PV_VALID(count_rows(query, "layers"), row_count);
	PV_VALID(count_rows(query, "selection"), row_count);

	// --- A selection moves under the same query object ------------------------
	Squey::PVSelection three(row_count);
	three.select_none();
	for (PVRow row = 0; row < 3; ++row) {
		three.set_line(row, true);
	}
	view->set_selection_view(three);

	PV_VALID(count_rows(query, "selection"), size_t(3));
	// A selection is not a layer: neither the stack nor the base layer moved.
	PV_VALID(count_rows(query, "layers"), row_count);
	PV_VALID(count_rows(query, "layer('All events')"), row_count);

	// The same holds for a query built from the source rather than from a view:
	// it resolves through whichever view is current, so the scopes still name
	// something. A caller with no view at all is the case where they widen to
	// every row, and that one has no selection to disagree with.
	{
		Squey::PVDuckDBQuery from_source(view->get_parent<Squey::PVSource>());
		PV_VALID(count_rows(from_source, "selection"), size_t(3));
		PV_VALID(count_rows(from_source, "layer('All events')"), row_count);
	}

	// --- A layer is reachable by the name it carries --------------------------
	view->commit_selection_to_new_layer("Three rows");
	PV_VALID(count_rows(query, "layer('Three rows')"), size_t(3));
	PV_VALID(count_rows(query, "layer('All events')"), row_count);

	// --- "layers" is what the stack lets through, read per query --------------
	set_visible(view, "All events", false);
	set_visible(view, "Three rows", true);
	PV_VALID(count_rows(query, "layers"), size_t(3));

	set_visible(view, "All events", true);
	PV_VALID(count_rows(query, "layers"), row_count);
	// Hiding never touched the layer itself.
	PV_VALID(count_rows(query, "layer('Three rows')"), size_t(3));

	// --- A misnamed layer is an error, not an empty result --------------------
	// An empty result would read like an answer, and the name was typed by hand,
	// so the error says what it could have meant.
	{
		bool threw = false;
		try {
			query.run_tabular("SELECT COUNT(*) FROM layer('Three rrows')");
		} catch (const std::exception& e) {
			threw = true;
			const std::string message = e.what();
			PV_ASSERT_VALID(message.find("no layer named 'Three rrows'") != std::string::npos,
			                "the error names what was asked for", message);
			PV_ASSERT_VALID(message.find("'Three rows'") != std::string::npos,
			                "the error lists the layers that exist", message);
		}
		PV_ASSERT_VALID(threw, "a misnamed layer must be an error", threw);
	}

	// --- The text form shows a cell the format could not read -----------------
	// The oracle is pvcop's own rendering, which is what the listing draws.
	{
		Squey::PVSelection everything(row_count);
		everything.select_all();
		view->set_selection_view(everything);

		const pvcop::db::array& numbers = nraw.column(NUMBER_COL);
		const auto shown = query.run_tabular(
		    "SELECT " + num + " FROM layer('All events', text := true) ORDER BY rowid", nullptr,
		    row_count);
		PV_VALID(shown.rows.size(), row_count);
		PV_VALID(shown.column_types[0], std::string("VARCHAR"));

		size_t unreadable = 0;
		for (size_t row = 0; row < row_count; ++row) {
			PV_VALID(shown.rows[row][0], numbers.at(row));
			unreadable += static_cast<size_t>(not numbers.is_valid(row));
		}
		PV_ASSERT_VALID(unreadable > 0, "the file must hold unreadable cells", unreadable);

		// Which is a form the typed scope cannot produce: it has no number to
		// show for those rows, so it says NULL.
		const auto typed =
		    query.run_tabular("SELECT COUNT(" + num + ") FROM layer('All events')");
		PV_VALID(size_t(std::stoull(typed.rows[0][0])), row_count - unreadable);

		// And it is a value a query can be written about. Nothing is pushed down
		// in text mode -- what pvcop would compare against is the stored encoding
		// rather than this text -- so this is DuckDB filtering the emitted text.
		PV_VALID(count_rows(query, "layer('All events', text := true) WHERE " + num + " = 'test1'"),
		         size_t(1));
		PV_VALID(count_rows(query, "layer('All events', text := true) WHERE " + num + " = '1'"),
		         size_t(1));
	}

	// --- A textual column reads the same either way ---------------------------
	// Its cells are already the text they were written with, so the mode changes
	// nothing about them -- including the empty ones, which stay empty strings
	// rather than becoming NULL.
	{
		const auto plain =
		    query.run_tabular("SELECT " + txt + " FROM selection ORDER BY rowid", nullptr, row_count);
		const auto as_text = query.run_tabular(
		    "SELECT " + txt + " FROM selection(text := true) ORDER BY rowid", nullptr, row_count);
		PV_VALID(plain.rows.size(), row_count);
		PV_VALID(as_text.rows.size(), row_count);
		for (size_t row = 0; row < row_count; ++row) {
			PV_VALID(as_text.rows[row][0], plain.rows[row][0]);
		}

		// Which is why such a column keeps the pvcop shortcut in text mode: the
		// storage already holds the text a filter compares against.
		PV_VALID(count_rows(query, "selection(text := true) WHERE " + txt + " = 'test1'"),
		         size_t(1));
		PV_VALID(count_rows(query, "selection WHERE " + txt + " = 'test1'"), size_t(1));
	}

	return 0;
}
