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

// A pushed filter carries its constant as a DuckDB Value; answering it with
// pvcop means writing that value as text and having pvcop parse it back. On a
// text or an integer column the round trip is faithful. On a business type it
// is not: an IPv6 column is exposed to SQL as a 128-bit integer, so the constant
// in "col = 42535295865117307932921825928971026434" is a number, while pvcop's
// parser for that column expects "2001:400:0:69::2".
//
// Feeding one to the other must not produce a selection at all -- neither a
// wrong one nor an empty one passed off as an answer. What this test pins is
// that the result matches pvcop read directly, whichever route the scan chose.
//
// Run once per business type, taking the file, format and column from argv, the
// way the ordering test does.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"

int main(int argc, char** argv)
{
	PV_ASSERT_VALID(argc >= 4, "usage", std::string("<file> <format> <column_name>"));
	const std::string file = argv[1];
	const std::string format = argv[2];
	const std::string column_name = argv[3];

	pvtest::TestEnv env(file, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	// --- No ordinary query may trip the optional-filter guard -----------------
	// The scan refuses an optional filter of a kind nobody has vetted, which is
	// a tripwire meant to fire on a DuckDB upgrade -- but a whitelist is only
	// worth the shapes it has been held against, and a user typing one of these
	// would meet the refusal rather than an answer. So they run here, on the
	// column where pvcop conversion fails and filters are therefore dropped
	// rather than answered.
	//
	// A failure is not a bug to paper over: it means DuckDB now pushes a kind
	// nobody has looked at, and the answer is to look at it and either vet it in
	// vetted_optional_kind() or apply it.
	{
		const std::string c = Squey::PVDuckDBQuery::quote_identifier(argv[3]);
		const std::string shapes[] = {
		    "SELECT rowid FROM layers WHERE " + c + " IN (SELECT " + c + " FROM layers LIMIT 5)",
		    "SELECT rowid FROM layers WHERE " + c + " NOT IN (SELECT " + c + " FROM layers LIMIT 5)",
		    "SELECT rowid FROM layers WHERE EXISTS (SELECT 1 FROM layers t WHERE t.rowid = layers.rowid)",
		    "SELECT rowid FROM layers ORDER BY " + c + " LIMIT 5",
		    "SELECT rowid FROM layers a JOIN layers b USING (rowid) LIMIT 5",
		    "SELECT rowid FROM layers WHERE " + c + " IS NULL",
		    "SELECT rowid FROM layers WHERE " + c + " IS NOT NULL",
		    "SELECT rowid FROM layers WHERE rowid BETWEEN 1 AND 5",
		    "SELECT rowid FROM layers WHERE rowid = 1 AND " + c + " IS NOT NULL",
		    "SELECT rowid FROM layers WHERE rowid NOT IN (1, 2, 3)",
		    "SELECT rowid FROM layers WHERE rowid < 5 OR rowid > 9000",
		    "SELECT rowid FROM layers QUALIFY row_number() OVER () < 5",
		    "SELECT rowid FROM layers WHERE rowid IN (SELECT MAX(rowid) FROM layers)",
		    "SELECT rowid FROM layers SEMI JOIN (SELECT 1 AS rowid) USING (rowid)",
		};
		for (const std::string& sql : shapes) {
			std::string failure;
			try {
				query.run_tabular(sql, nullptr, 2);
			} catch (const std::exception& e) {
				failure = e.what();
			}
			PV_ASSERT_VALID(failure.empty(), "query refused", sql, "error", failure);
		}

		// And the session survived every one of them. A refusal raised as an
		// internal error would have invalidated the database instead, taking the
		// console down with it for the rest of the session -- which is what the
		// first version of the guard did.
		const auto still_there = query.run_tabular("SELECT COUNT(*) FROM layers", nullptr, 1);
		PV_VALID(size_t(std::stoull(still_there.rows[0][0])), row_count);
	}

	// The column is found by name rather than by index, so the same test serves
	// formats whose axes are ordered differently.
	const std::vector<std::string> names = query.column_names();
	size_t index = names.size();
	for (size_t i = 1; i < names.size(); ++i) {
		if (names[i] == column_name) {
			index = i;
			break;
		}
	}
	PV_ASSERT_VALID(index < names.size(), "column not found", column_name);

	const PVCol col(static_cast<PVCol::value_type>(index - 1));
	const pvcop::db::array& array = nraw.column(col);
	const std::string quoted = Squey::PVDuckDBQuery::quote_identifier(column_name);
	const std::string sql_type = query.column_types()[index];

	Squey::PVSelection out(row_count);

	// --- Equality against the value SQL actually sees --------------------------
	// Read the stored value the way SQL reads it, then ask for it back. This is
	// the query a user writes after looking at the listing, and the one where a
	// constant handed to the wrong parser would quietly select nothing.
	{
		const auto shown = query.run_tabular(
		    "SELECT " + quoted + " FROM layers WHERE rowid = 0", nullptr, 1);
		PV_VALID(shown.rows.size(), size_t(1));
		const std::string literal = shown.rows[0][0];

		// The reference is pvcop: whatever the scan did, the rows selected must
		// be those holding the same value as row 0.
		const std::string wanted = array.at(0);
		size_t expected = 0;
		for (size_t row = 0; row < row_count; ++row) {
			expected += static_cast<size_t>(array.at(row) == wanted);
		}
		PV_ASSERT_VALID(expected > 0, "expected", expected);

		// Numeric types take the literal bare, text types in quotes.
		const bool numeric = sql_type != "VARCHAR";
		const std::string predicate =
		    quoted + " = " + (numeric ? literal : "'" + literal + "'");

		query.select("SELECT rowid FROM layers WHERE " + predicate, out);
		PV_ASSERT_VALID(out.bit_count() == expected, "type", sql_type, "predicate", predicate,
		                "sql", out.bit_count(), "pvcop", expected);
	}

	// --- Membership over three stored values -----------------------------------
	{
		const auto shown = query.run_tabular(
		    "SELECT " + quoted + " FROM layers WHERE rowid IN (0, 1, 2) ORDER BY rowid", nullptr,
		    3);
		PV_VALID(shown.rows.size(), size_t(3));

		size_t expected = 0;
		for (size_t row = 0; row < row_count; ++row) {
			const std::string value = array.at(row);
			expected += static_cast<size_t>(value == array.at(0) || value == array.at(1) ||
			                                value == array.at(2));
		}

		const bool numeric = sql_type != "VARCHAR";
		std::string list;
		for (const auto& r : shown.rows) {
			if (not list.empty()) {
				list += ", ";
			}
			list += numeric ? r[0] : "'" + r[0] + "'";
		}

		query.select("SELECT rowid FROM layers WHERE " + quoted + " IN (" + list + ")", out);
		PV_ASSERT_VALID(out.bit_count() == expected, "type", sql_type, "list", list, "sql",
		                out.bit_count(), "pvcop", expected);
	}

	// --- The round trip has to hold for every value, not for one ---------------
	// A parser can read one spelling and not another -- an IPv6 written "::2"
	// against the same address written out in full, say -- so the check runs over
	// many rows rather than one.
	{
		const size_t probes = std::min<size_t>(64, row_count);
		const bool numeric = sql_type != "VARCHAR";

		for (size_t row = 0; row < probes; ++row) {
			const auto shown = query.run_tabular(
			    "SELECT " + quoted + " FROM layers WHERE rowid = " + std::to_string(row), nullptr, 1);
			const std::string literal = numeric ? shown.rows[0][0] : "'" + shown.rows[0][0] + "'";

			const std::string wanted = array.at(row);
			size_t expected = 0;
			for (size_t r = 0; r < row_count; ++r) {
				expected += static_cast<size_t>(array.at(r) == wanted);
			}

			query.select("SELECT rowid FROM layers WHERE " + quoted + " = " + literal, out);
			PV_ASSERT_VALID(out.bit_count() == expected, "row", row, "type", sql_type, "literal",
			                literal, "sql", out.bit_count(), "pvcop", expected);
		}
	}

	return 0;
}
