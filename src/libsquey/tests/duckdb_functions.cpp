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

// What a query can call is DuckDB's, and there are hundreds of them, so a
// console cannot print the list and a user cannot be expected to know it. The
// completion offers them instead, which means the catalogue has to come back as
// something one can put in a list: a name per line, spelt the way it is typed.
//
// What is checked here is that filter. The catalogue also holds the operators,
// under the names they are written with, and the table functions, which are not
// called from where a value goes -- both would be offered where nothing could
// use them.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <algorithm>
#include <set>
#include <string>

#include "common.h"

const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVDuckDBQuery query(*env.root.current_view());

	const auto listed = query.functions();

	// A range rather than a number: the catalogue grows with DuckDB, and what
	// this says is that the filter neither emptied it nor let it through whole.
	PV_ASSERT_VALID(listed.size() > 200 && listed.size() < 2000,
	                "the function list is not a list of functions", listed.size());

	std::set<std::string> names;
	size_t described = 0;
	for (const auto& [name, description] : listed) {
		// Spelt the way it is typed: the catalogue lists the operators under
		// their own names -- "&&", "%", "!__postfix" -- and no prefix a user
		// types reaches those.
		PV_ASSERT_VALID(not name.empty() && (name[0] >= 'a' && name[0] <= 'z'),
		                "a name no keystroke reaches", name);
		for (const char c : name) {
			PV_ASSERT_VALID((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_',
			                "a name no keystroke reaches", name);
		}
		// The catalogue's own readers are not what a query is written with.
		PV_ASSERT_VALID(name.rfind("duckdb_", 0) != 0, "a catalogue reader was offered", name);
		// One entry per name: overloads differ by their arguments, which a list
		// of names has nowhere to show, so several rows would read as repeats.
		PV_ASSERT_VALID(names.insert(name).second, "the same function twice", name);
		described += size_t(not description.empty());
	}

	// Ordered, since the popup shows them in the order they arrive.
	PV_ASSERT_VALID(std::is_sorted(listed.begin(), listed.end()),
	                "the functions came back in no order", 0);

	// The ones a query is actually written with.
	for (const char* expected : {"upper", "lower", "count", "sum", "regexp_matches", "strlen"}) {
		PV_ASSERT_VALID(names.count(expected) == 1, "missing from the offered functions",
		                expected);
	}
	// And not the table functions, which go after FROM rather than where a
	// value does. Not "range", which is both -- it also builds a list, and that
	// one is called where a value goes.
	for (const char* table_only : {"read_csv", "read_parquet", "glob"}) {
		PV_ASSERT_VALID(names.count(table_only) == 0, "a table function was offered", table_only);
	}

	// Most carry DuckDB's own sentence, which is what the popup shows beside
	// the name. Not all do, and one that does not is still worth offering.
	PV_ASSERT_VALID(described * 10 > listed.size() * 8, "the descriptions went missing",
	                described, "of", listed.size());

	return 0;
}
