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

// A cell the format could not read still occupies a slot in the storage, and
// what sits there is an internal encoding: an empty cell of a number column
// reads back as 0, and "test1" as 1. Read straight, SQL would compare and sum
// those as if they were data -- and the answer would depend on who evaluated
// the predicate, since pvcop excludes such rows and DuckDB, given the raw
// storage, does not.
//
// The scan therefore emits them as NULL, which is what the rest of the
// application already does with them: its search filter never matches one
// against a valid literal, and offers "valid / invalid / all values" as the
// choice that IS NULL and IS NOT NULL express in SQL.
//
// The filter is the oracle here rather than pvcop read directly: what is being
// pinned is agreement with the application, not with a library.
//
// A textual column is deliberately left alone, and checked here too: pvcop
// hands back its cells as they were written, empty string included, so there is
// no encoding to hide and nothing to turn into NULL.

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
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "common.h"

// Thirteen rows over two axes holding the same text: six readable numbers, four
// words and three empty cells. col1 is typed as a number, so its unreadable
// cells fall back on the storage encoding; col2 is text, where they do not.
const std::string filename = TEST_FOLDER "/picviz/errors_search.csv";
const std::string fileformat = TEST_FOLDER "/picviz/errors_search.csv.format";

constexpr PVCol NUMBER_COL(0);
constexpr PVCol TEXT_COL(1);

/**
 * Rows the multiple-search filter selects for @a value on @a col, in the mode
 * the listing's context menu uses -- every value, readable or not.
 */
static size_t search_count(Squey::PVView* view, PVCol col, const std::string& value)
{
	constexpr char plugin_name[] = "search-multiple";
	Squey::PVLayerFilter::p_type filter =
	    LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name(plugin_name)
	        ->clone<Squey::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter(plugin_name);

	Squey::PVLayer out("Out", view->get_row_count());
	out.reset_to_empty_and_default_color();
	Squey::PVLayer& in = view->get_layer_stack_output_layer();
	filter->set_view(view);
	filter->set_output(&out);

	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(col));
	// include, match case, the entire field, plain text, all values
	const std::pair<const char*, int> enums[] = {
	    {"include", 0}, {"case", 1}, {"entire", 1}, {"interpret", 0}, {"type", 2}};
	for (const auto& [key, sel] : enums) {
		auto e = args[key].value<PVCore::PVEnumType>();
		e.set_sel(sel);
		args[key].setValue(e);
	}
	args["exps"].setValue(PVCore::PVPlainTextType(value.c_str()));
	filter->set_args(args);
	filter->operator()(in);

	return out.get_selection().bit_count();
}

static size_t count_rows(const Squey::PVDuckDBQuery& query, const std::string& predicate)
{
	Squey::PVSelection out(0);
	const auto table = query.run_tabular("SELECT COUNT(*) FROM layers WHERE " + predicate);
	return std::stoull(table.rows[0][0]);
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	const pvcop::db::array& numbers = nraw.column(NUMBER_COL);
	const pvcop::db::array& text = nraw.column(TEXT_COL);

	size_t unreadable = 0;
	for (size_t row = 0; row < row_count; ++row) {
		unreadable += static_cast<size_t>(not numbers.is_valid(row));
	}
	PV_ASSERT_VALID(unreadable > 0, "the file must hold unreadable cells", unreadable);

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());
	const std::string num = Squey::PVDuckDBQuery::quote_identifier(query.column_names()[1]);
	const std::string txt = Squey::PVDuckDBQuery::quote_identifier(query.column_names()[2]);

	// --- The answer must not depend on how the predicate is written -----------
	// These three select the same rows. They used to give three different counts:
	// pvcop answered the first two and DuckDB the third, on the raw storage.
	{
		const size_t equality = count_rows(query, num + " = 0");
		const size_t disjunction = count_rows(query, num + " = 0 OR " + num + " = 1 OR " + num + " = 2");
		const size_t membership = count_rows(query, num + " IN (0, 1, 2)");

		PV_VALID(equality, search_count(view, NUMBER_COL, "0"));
		PV_VALID(disjunction, membership);
		PV_VALID(membership, search_count(view, NUMBER_COL, "0\n1\n2"));
	}

	// --- The application's three modes, spelled in SQL -------------------------
	{
		PV_VALID(count_rows(query, num + " IS NULL"), unreadable);
		PV_VALID(count_rows(query, num + " IS NOT NULL"), row_count - unreadable);

		// Aggregates follow: an unreadable cell is not a zero to be averaged in.
		const auto counted = query.run_tabular("SELECT COUNT(" + num + ") FROM layers");
		PV_VALID(size_t(std::stoull(counted.rows[0][0])), row_count - unreadable);
	}

	// --- Grouping reports what the file holds ----------------------------------
	// Every readable value once, and one bucket for everything unreadable. The
	// encoding used to show up here as buckets for values the file never held.
	{
		const auto groups = query.run_tabular("SELECT " + num + ", COUNT(*) FROM layers GROUP BY 1");
		size_t counted = 0;
		for (const auto& row : groups.rows) {
			// run_tabular renders NULL as an empty cell.
			counted += row[0].empty() ? 0 : std::stoull(row[1]);
			if (row[0].empty()) {
				PV_VALID(size_t(std::stoull(row[1])), unreadable);
			}
		}
		PV_VALID(counted, row_count - unreadable);
	}

	// --- The textual column keeps its cells ------------------------------------
	// An empty cell of a text column is an empty string, which the listing shows
	// and the search finds. Turning it into NULL would part company with both.
	{
		size_t empty_cells = 0;
		for (size_t row = 0; row < row_count; ++row) {
			empty_cells += static_cast<size_t>(text.at(row).empty());
		}
		PV_ASSERT_VALID(empty_cells > 0, "the file must hold empty text cells", empty_cells);

		PV_VALID(count_rows(query, txt + " = ''"), empty_cells);
		PV_VALID(count_rows(query, txt + " IS NULL"), size_t(0));
	}

	// --- An address, which cannot be written the way the column stores it ------
	// The scan is handed its constant as the number the column holds -- DuckDB
	// writes 10.0.0.1 as 167772161 -- and an address type cannot read that back.
	// Failing to convert that way used to pass for a literal that had converted
	// into an unreadable value, and pvcop matches those against the rows that are
	// themselves unreadable: a search for an address answered with every row that
	// had no address at all, for any address asked for.
	//
	// A number column cannot show this, its literals being written the same way
	// either side, which is why everything above ran on one.
	{
		const std::string addresses = pvtest::get_tmp_filename() + ".csv";
		const std::string addresses_format = addresses + ".format";
		{
			std::ofstream(addresses) << "a,10.0.0.1\nb,\nc,10.0.0.1\nd,10.0.0.2\n";
			std::ofstream(addresses_format)
			    << R"(<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE PVParamXml>)"
			       R"(<param version="9" first_line="0"><splitter type="csv" sep=",">)"
			       R"(<field><axis name="who" type="string"/></field>)"
			       R"(<field><axis name="addr" type="ipv4"/></field>)"
			       R"(</splitter></param>)";
		}

		Squey::PVSource& source =
		    env.add_source(std::vector<std::string>{addresses}, addresses_format, 1, false);
		env.compute_mapping(0, 1);
		env.compute_scaling(0, 1, 0).emplace_add_child();

		Squey::PVDuckDBQuery addressed(source);
		PV_VALID(count_rows(addressed, "addr IS NULL"), size_t(1));
		PV_VALID(count_rows(addressed, "addr = ipv4('10.0.0.1')"), size_t(2));
		PV_VALID(count_rows(addressed, "addr = ipv4('10.0.0.2')"), size_t(1));
		// An address the file does not hold matches nothing. It used to answer
		// with the unreadable rows, which is how the whole thing came to light.
		PV_VALID(count_rows(addressed, "addr = ipv4('10.0.0.9')"), size_t(0));
		// The same three ways of spelling a membership as above.
		PV_VALID(count_rows(addressed, "addr = ipv4('10.0.0.1') OR addr = ipv4('10.0.0.2')"),
		         size_t(3));
		PV_VALID(count_rows(addressed, "addr IN (ipv4('10.0.0.1'), ipv4('10.0.0.2'))"), size_t(3));

		std::remove(addresses.c_str());
		std::remove(addresses_format.c_str());
	}

	return 0;
}
