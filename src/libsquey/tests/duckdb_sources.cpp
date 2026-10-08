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

// A query reaches sources other than the one its console sits on, so that a
// join can cross them -- what a Python script calling squey.source() could
// always do and SQL could not.
//
// Two forms say the same thing: an argument, "selection(source := 'name')",
// which always works, and a schema, "name.selection", which reads better but
// only exists where the name is unique. Source names are not unique -- the
// Python API indexes them by (name, position) for that reason -- so what is
// checked here is that the argument form survives namesakes, that the sugar is
// withheld exactly where it could not tell them apart, and that naming a source
// that is not there is an error rather than an empty answer.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"

// Two different files, so that a row count tells which source answered, and a
// third import of the first one, whose name therefore collides with it.
const std::string file_a = TEST_FOLDER "/picviz/errors_search.csv";
const std::string format_a = TEST_FOLDER "/picviz/errors_search.csv.format";
const std::string file_b = TEST_FOLDER "/picviz/ipv4_default_mapping.csv";
const std::string format_b = TEST_FOLDER "/picviz/ipv4_default_mapping.csv.format";

static size_t count_rows(const Squey::PVDuckDBQuery& query, const std::string& from)
{
	const auto table = query.run_tabular("SELECT COUNT(*) FROM " + from);
	PV_ASSERT_VALID(table.rows.size() == 1, "one row of count", from);
	return size_t(std::stoull(table.rows[0][0]));
}

//! Whether @a sql fails, which is how a query naming something absent must end.
static bool fails(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	try {
		query.run_tabular(sql);
	} catch (const std::runtime_error&) {
		return true;
	}
	return false;
}

int main()
{
	pvtest::TestEnv env(file_a, format_a, 1, pvtest::ProcessUntil::View);
	// Same scene: a project holds its sources side by side, and that is what a
	// console sees.
	Squey::PVSource& source_b = env.add_source(file_b, format_b, 1, /*new_scene=*/false);
	Squey::PVSource& twin = env.add_source(file_a, format_a, 1, /*new_scene=*/false);

	Squey::PVView* view = env.root.current_view();
	const size_t rows_a = view->get_rushnraw_parent().row_count();
	const size_t rows_b = source_b.get_rushnraw().row_count();
	// The join below counts on the two differing, otherwise a wrong source
	// would answer with the right number.
	PV_ASSERT_VALID(rows_a != rows_b, "rows_a", rows_a, "rows_b", rows_b);
	PV_VALID(size_t(twin.get_rushnraw().row_count()), rows_a);

	const std::string name_a = view->get_parent<Squey::PVSource>().get_name();
	const std::string name_b = source_b.get_name();
	PV_ASSERT_VALID(name_a != name_b, "name_a", name_a, "name_b", name_b);
	PV_VALID(twin.get_name(), name_a); // same file, hence the namesake

	Squey::PVDuckDBQuery query(*view);

	// --- The bare name still means the console's own source -------------------
	// Every query written before sources could be named has to keep answering
	// what it answered.
	PV_VALID(count_rows(query, "selection"), rows_a);
	PV_VALID(count_rows(query, "layers"), rows_a);

	// --- A named source answers for itself ------------------------------------
	const std::string b_literal = "'" + name_b + "'";
	PV_VALID(count_rows(query, "selection(source := " + b_literal + ")"), rows_b);
	PV_VALID(count_rows(query, "layers(source := " + b_literal + ")"), rows_b);

	// --- A join crosses them ---------------------------------------------------
	// On rowid, so that what is checked is the crossing itself rather than
	// anything the two files happen to hold in common.
	const auto joined = query.run_tabular("SELECT COUNT(*) FROM selection a "
	                                      "JOIN selection(source := " +
	                                      b_literal + ") b ON a.rowid = b.rowid");
	PV_VALID(size_t(std::stoull(joined.rows[0][0])), std::min(rows_a, rows_b));

	// --- The sugar says the same thing ----------------------------------------
	const std::string schema_b = Squey::PVDuckDBQuery::quote_identifier(name_b);
	PV_VALID(count_rows(query, schema_b + ".selection"), rows_b);
	PV_VALID(count_rows(query, schema_b + ".layers"), rows_b);

	// --- Namesakes are told apart by position, and only that way ---------------
	// Both carry the name of the first file, so the schema form cannot mean
	// either of them and must not exist -- a schema that silently picked one
	// would read like an answer.
	const std::string a_literal = "'" + name_a + "'";
	PV_VALID(count_rows(query, "selection(source := " + a_literal + ")"), rows_a);
	PV_VALID(count_rows(query, "selection(source := " + a_literal + ", source_position := 1)"), rows_a);
	PV_ASSERT_VALID(fails(query, "SELECT COUNT(*) FROM " +
	                                Squey::PVDuckDBQuery::quote_identifier(name_a) + ".selection"),
	                "a namesake gets no schema", name_a);

	// --- What is not there is an error, not an empty result --------------------
	PV_ASSERT_VALID(fails(query, "SELECT COUNT(*) FROM selection(source := 'no such source')"),
	                "refused", "an unknown source");
	PV_ASSERT_VALID(fails(query, "SELECT COUNT(*) FROM selection(source := " + a_literal +
	                                 ", source_position := 7)"),
	                "refused", "a position past the namesakes");
	PV_ASSERT_VALID(fails(query, "SELECT COUNT(*) FROM selection(source_position := 1)"),
	                "refused", "a position with no source to disambiguate");

	// --- The listing is how the names are found --------------------------------
	// Without it a query can only name a source it already knows, whereas a
	// Python script loops over them.
	const auto listing = query.run_tabular("SELECT name, position, current FROM sources "
	                                       "ORDER BY name, position");
	PV_VALID(listing.rows.size(), size_t(3));
	const auto current = query.run_tabular("SELECT name FROM sources WHERE current");
	PV_VALID(current.rows.size(), size_t(1));
	PV_VALID(current.rows[0][0], name_a);

	// --- A selection still comes from the console's own source ------------------
	// The join reaches across, but what it projects is a row of the view being
	// filtered: any other rowid would name a row of a listing nobody is looking
	// at.
	Squey::PVSelection out(rows_a);
	query.select("SELECT a.rowid FROM selection a JOIN selection(source := " + b_literal +
	                 ") b ON a.rowid = b.rowid",
	             out);
	PV_VALID(out.bit_count(), std::min(rows_a, rows_b));

	return 0;
}
