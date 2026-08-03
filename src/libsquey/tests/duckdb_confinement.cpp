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

// A query is written about the source the console was opened on, and needs
// nothing else. DuckDB opens rather more than that by default -- any file by
// path, other databases, extensions fetched over the network -- and what a
// console runs is text somebody typed.
//
// What is pinned here is that those doors are shut, and that shutting them left
// every reading form working. Two of these were open until this test was
// written: a file could be read by path, and a second statement could ride along
// after a semicolon, since the check that let a statement through only ever read
// its first keyword.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <string>
#include <vector>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

/**
 * Run @a sql and return the error it raised, or an empty string when it ran.
 */
static std::string refusal(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	try {
		query.run_tabular(sql, nullptr, 2);
		return {};
	} catch (const std::exception& e) {
		return e.what();
	}
}

static void must_refuse(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	PV_ASSERT_VALID(not refusal(query, sql).empty(), "went through", sql);
}

static void must_run(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	PV_ASSERT_VALID(refusal(query, sql).empty(), "was refused", sql, "error",
	                refusal(query, sql));
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const size_t row_count = view->get_rushnraw_parent().row_count();

	Squey::PVDuckDBQuery query(*view);

	// --- The filesystem is not reachable --------------------------------------
	// A path that exists, so a refusal is a refusal rather than a missing file.
	{
		const std::string self = "'" + filename + "'";
		must_refuse(query, "SELECT * FROM read_csv(" + self + ")");
		must_refuse(query, "SELECT * FROM read_text(" + self + ")");
		must_refuse(query, "SELECT * FROM read_blob(" + self + ")");
		must_refuse(query, "SELECT * FROM glob('/etc/*')");
		must_refuse(query, "SELECT * FROM sniff_csv(" + self + ")");
		must_refuse(query, "ATTACH '/tmp/squey_confinement_probe.db' AS probe");
		must_refuse(query, "COPY (SELECT 1) TO '/tmp/squey_confinement_probe.csv'");
	}

	// --- And cannot be reached again ------------------------------------------
	// The settings are locked, so a query cannot undo them.
	{
		must_refuse(query, "SET enable_external_access = true");
		must_refuse(query, "PRAGMA enable_external_access = true");
		// Which the setting itself confirms, read through a query.
		const auto shown = query.run_tabular(
		    "SELECT value FROM duckdb_settings() WHERE name = 'enable_external_access'", nullptr, 1);
		PV_VALID(shown.rows.size(), size_t(1));
		PV_VALID(shown.rows[0][0], std::string("false"));
	}

	// --- Nothing that changes anything ----------------------------------------
	{
		must_refuse(query, "CREATE TABLE scratch AS SELECT 1");
		must_refuse(query, "DROP VIEW layers");
		must_refuse(query, "CREATE OR REPLACE VIEW layers AS SELECT 1 AS rowid");
		must_refuse(query, "INSTALL httpfs");
		must_refuse(query, "DELETE FROM layers");
	}

	// --- A second statement does not ride along -------------------------------
	// DuckDB runs every statement a string holds, so reading the first keyword
	// let this through and the view was gone afterwards.
	{
		const std::string error = refusal(query, "SELECT 1; DROP VIEW layers");
		PV_ASSERT_VALID(not error.empty(), "a second statement went through", 0);
		PV_ASSERT_VALID(error.find("one statement at a time") != std::string::npos,
		                "the error does not say what is wrong", error);
		must_refuse(query, "SELECT 1; SELECT 2");
	}

	// --- Everything the console offers still runs ------------------------------
	// The gate reads the parsed statement rather than a keyword, so this is what
	// says it did not narrow what a query may ask.
	{
		must_run(query, "SELECT COUNT(*) FROM layers");
		must_run(query, "WITH x AS (SELECT rowid FROM layers) SELECT COUNT(*) FROM x");
		must_run(query, "FROM layers LIMIT 1");
		must_run(query, "VALUES (1)");
		must_run(query, "TABLE layers");
		must_run(query, "SHOW TABLES");
		must_run(query, "DESCRIBE layers");
		must_run(query, "EXPLAIN SELECT rowid FROM layers");
		must_run(query, "PRAGMA version");
		must_run(query, "SUMMARIZE layers");
		// And a bare predicate, which is the form the console teaches.
		must_run(query, "rowid < 3");
	}

	// --- The source is still there --------------------------------------------
	// Every refusal above had to leave the catalog as it was; a DROP that ran
	// before being reported would pass every assertion up to here.
	{
		const auto shown = query.run_tabular("SELECT COUNT(*) FROM layers", nullptr, 1);
		PV_VALID(size_t(std::stoull(shown.rows[0][0])), row_count);
		PV_ASSERT_VALID(query.column_names().size() > 1, "the schema is gone",
		                query.column_names().size());
	}

	return 0;
}
