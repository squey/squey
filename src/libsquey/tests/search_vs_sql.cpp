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

// Times the same search run two ways: through the multiple-search layer filter,
// and through the SQL console's query engine.
//
// The two are only comparable if they select the same rows, so every case is
// checked for equality before its timings are reported -- a search that answers
// something else is not faster, it is wrong. That check is also what makes this
// runnable as a test: at its default size it is an agreement oracle between the
// two engines, and the timings are what one reads when running it by hand on a
// larger dataset.
//
// Usage: SQUEY_TEST_Tsquey_search_vs_sql [duplication_factor] [repeats]
//
// Two phases, on datasets sized to the same row count:
//
//   - a text axis holding random 10-character values, all distinct. That is the
//     shape a search actually meets -- a column whose dictionary is as large as
//     the column itself -- and it is deliberately hostile to any strategy that
//     leans on repeated values. SQL has no layout it can read directly here, so
//     the scan builds every string;
//
//   - an integer axis, which the scan hands to SQL as a pointer into the nraw.
//     The same exact-match query then measures the engines rather than the cost
//     of materialising text, which is what makes the first phase's numbers mean
//     something.

#include "common.h"

#include <squey/PVDuckDBQuery.h>
#include <squey/PVLayer.h>
#include <squey/PVLayerFilter.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>
#include <pvkernel/core/squey_assert.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr char TEXT_FILE[] = TEST_FOLDER "/picviz/axes_types_discovery.csv";
constexpr char TEXT_FORMAT[] = TEST_FOLDER "/picviz/axes_types_discovery.csv.format";
//! The axis holding random text, and its name once exposed to SQL.
constexpr PVCol TEXT_COL(13);
constexpr char TEXT_COL_NAME[] = "string";
//! A value of that column, so the exact-match cases have something to find.
constexpr char KNOWN_VALUE[] = "cbacbabacabcabc";

constexpr char NUMBER_FILE[] = TEST_FOLDER "/picviz/heat_line.csv";
constexpr char NUMBER_FORMAT[] = TEST_FOLDER "/picviz/heat_line.csv.format";
constexpr PVCol NUMBER_COL(1);
constexpr char NUMBER_COL_NAME[] = "uint8";

//! The number file is a quarter the length of the text one, so both phases run
//! over the same row count for a given duplication factor.
constexpr size_t NUMBER_DUP_RATIO = 4;

/**
 * The filter's six enum arguments, in the order its UI presents them.
 *
 * Only the last four vary here: the axis is always the text one, and the search
 * always includes rather than excludes.
 */
struct SearchOptions {
	int include;   //!< 0 include, 1 exclude
	int case_;     //!< 0 insensitive, 1 sensitive
	int entire;    //!< 0 part of the field, 1 the entire field
	int interpret; //!< 0 plain text, 1 regular expression
	int type;      //!< 0 valid, 1 invalid, 2 all values
};

struct Case {
	const char* name;
	SearchOptions options;
	//! Expressions, one per line, as typed in the filter's text box.
	std::string expressions;
	//! The predicate that means the same thing to the SQL console.
	std::string predicate;
};

/**
 * Wall time of the fastest of @a repeats runs.
 *
 * The fastest rather than the mean: a slower run only ever means the machine was
 * busy elsewhere, so the minimum is the measurement least polluted by whatever
 * else the box is doing.
 */
template <class F>
double best_ms(size_t repeats, F&& run)
{
	double best = std::numeric_limits<double>::max();
	for (size_t i = 0; i < repeats; ++i) {
		const auto start = std::chrono::steady_clock::now();
		run();
		const std::chrono::duration<double, std::milli> elapsed =
		    std::chrono::steady_clock::now() - start;
		best = std::min(best, elapsed.count());
	}
	return best;
}

void set_args(PVCore::PVArgumentList& args, PVCol col, const Case& c)
{
	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(col));

	const std::pair<const char*, int> enums[] = {{"include", c.options.include},
	                                            {"case", c.options.case_},
	                                            {"entire", c.options.entire},
	                                            {"interpret", c.options.interpret},
	                                            {"type", c.options.type}};
	for (const auto& [key, value] : enums) {
		auto e = args[key].value<PVCore::PVEnumType>();
		e.set_sel(value);
		args[key].setValue(e);
	}

	args["exps"].setValue(PVCore::PVPlainTextType(c.expressions.c_str()));
}

/**
 * Run every case both ways on @a view, checking they agree before reporting.
 */
void run_phase(const char* title,
               Squey::PVView* view,
               PVCol col,
               const char* col_name,
               const std::vector<Case>& cases,
               size_t repeats)
{
	const size_t row_count = view->get_row_count();
	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	// State what is being measured: on a column SQL cannot read without building
	// each value, the comparison says something quite different than on one it
	// maps straight through.
	const std::vector<std::string> names = query.column_names();
	const std::vector<std::string> types = query.column_types();
	PV_VALID(names[col + 1], std::string(col_name));

	// The filter is driven exactly as the listing's context menu drives it.
	constexpr char plugin_name[] = "search-multiple";
	Squey::PVLayerFilter::p_type filter =
	    LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name(plugin_name)
	        ->clone<Squey::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter(plugin_name);

	Squey::PVLayer out("Out", row_count);
	out.reset_to_empty_and_default_color();
	Squey::PVLayer& in = view->get_layer_stack_output_layer();
	filter->set_view(view);
	filter->set_output(&out);

	std::cout << "\n" << title << ": rows=" << row_count << " column=" << col_name
	          << " sql_type=" << types[col + 1] << " repeats=" << repeats << "\n";
	std::cout << std::left << std::setw(24) << "case" << std::right << std::setw(12) << "filter ms"
	          << std::setw(12) << "sql ms" << std::setw(10) << "ratio" << std::setw(12) << "rows"
	          << "\n";

	for (const Case& c : cases) {
		set_args(args, col, c);
		filter->set_args(args);

		const double filter_ms = best_ms(repeats, [&]() { filter->operator()(in); });
		const Squey::PVSelection filter_sel = out.get_selection();

		Squey::PVSelection sql_sel(row_count);
		const double sql_ms = best_ms(repeats, [&]() { query.select(c.predicate, sql_sel); });

		// Timings of two different answers would be meaningless, so disagreement
		// is a failure rather than a footnote.
		PV_ASSERT_VALID((filter_sel ^ sql_sel).is_empty(), "case", std::string(c.name),
		                "filter_rows", filter_sel.bit_count(), "sql_rows", sql_sel.bit_count());

		std::cout << std::left << std::setw(24) << c.name << std::right << std::fixed
		          << std::setprecision(1) << std::setw(12) << filter_ms << std::setw(12) << sql_ms
		          << std::setw(9) << std::setprecision(2) << (sql_ms / filter_ms) << "x"
		          << std::setw(12) << filter_sel.bit_count() << "\n";
	}
}

} // namespace

int main(int argc, char** argv)
{
	const size_t dup = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 1;
	const size_t repeats = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 3;

	{
		const std::string col = Squey::PVDuckDBQuery::quote_identifier(TEXT_COL_NAME);

		// One case per search mode the filter offers, each paired with the SQL
		// that expresses the same thing. Selectivities differ on purpose: an
		// exact match finds one row per duplicate, a substring a few hundred.
		const std::vector<Case> cases = {
		    {"exact match", {0, 1, 1, 0, 2}, KNOWN_VALUE, col + " = '" + KNOWN_VALUE + "'"},
		    {"exact match, 3 values",
		     {0, 1, 1, 0, 2},
		     std::string(KNOWN_VALUE) + "\ncbabacabcabcbca\nbacabcabcbcacba",
		     col + " IN ('" + KNOWN_VALUE + "', 'cbabacabcabcbca', 'bacabcabcbcacba')"},
		    {"contains", {0, 1, 0, 0, 2}, "ab", col + " LIKE '%ab%'"},
		    {"contains, any case", {0, 0, 0, 0, 2}, "AB", col + " ILIKE '%AB%'"},
		    {"regular expression", {0, 1, 0, 1, 2}, "a.b", "regexp_matches(" + col + ", 'a.b')"},
		};

		pvtest::TestEnv env(TEXT_FILE, TEXT_FORMAT, dup, pvtest::ProcessUntil::View);
		run_phase("text axis", env.root.current_view(), TEXT_COL, TEXT_COL_NAME, cases, repeats);
	}

	{
		// Only equality survives the change of column: the filter has no notion
		// of a range, and a substring of a number is not what either engine is
		// asked for in practice.
		const std::string col = Squey::PVDuckDBQuery::quote_identifier(NUMBER_COL_NAME);
		const std::vector<Case> cases = {
		    {"exact match", {0, 1, 1, 0, 2}, "12345", col + " = 12345"},
		    {"exact match, 3 values", {0, 1, 1, 0, 2}, "12345\n23456\n34567",
		     col + " IN (12345, 23456, 34567)"},
		};

		pvtest::TestEnv env(NUMBER_FILE, NUMBER_FORMAT, dup * NUMBER_DUP_RATIO,
		                    pvtest::ProcessUntil::View);
		run_phase("integer axis", env.root.current_view(), NUMBER_COL, NUMBER_COL_NAME, cases,
		          repeats);
	}

	return 0;
}
