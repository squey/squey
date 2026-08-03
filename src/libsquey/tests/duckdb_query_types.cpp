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

// Covers the two column kinds the numeric test does not reach: an ipv4 column,
// which must keep pvcop's ordering rather than sort as text, and string columns,
// which fall back to their textual form.
//
// The ipv4 ordering check is the important one. pvcop stores an address as a
// host-order uint32 precisely so that comparing the integer compares the
// address; exposing it as VARCHAR instead would sort "10.x" between "1.x" and
// "2.x", and the mistake would be silent -- the queries would still return
// rows, just the wrong ones.

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <algorithm>
#include <string>
#include <vector>

#include "common.h"

const std::string filename = TEST_FOLDER "/sources/proxÿ.log";
const std::string fileformat = TEST_FOLDER "/formats/proxÿ.log.format";

static size_t count_selected(const PVCore::PVSelBitField& sel, size_t row_count)
{
	size_t count = 0;
	for (size_t row = 0; row < row_count; ++row) {
		count += static_cast<size_t>(sel.get_line(PVRow(row)));
	}
	return count;
}

int main()
{
	pvtest::TestEnv env(filename, fileformat, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	// Locate one ipv4 column and one string column, so the test does not depend
	// on the format's column order.
	PVCol ipv4_col(PVCol::value_type(-1));
	PVCol string_col(PVCol::value_type(-1));
	for (PVCol c(0); c < nraw.column_count(); ++c) {
		const std::string type = nraw.column(c).type();
		if (type == "ipv4" && ipv4_col == PVCol::value_type(-1)) {
			ipv4_col = c;
		}
		if (type == "string" && string_col == PVCol::value_type(-1)) {
			string_col = c;
		}
	}
	PV_ASSERT_VALID(ipv4_col != PVCol::value_type(-1), "ipv4_col", size_t(ipv4_col));
	PV_ASSERT_VALID(string_col != PVCol::value_type(-1), "string_col", size_t(string_col));

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());

	// Column names are the axis names verbatim, so they go through
	// quote_identifier() before being written into a query -- an axis named
	// "arp.duplicate-address-detected" is a valid identifier only when quoted.
	const std::vector<std::string> names = query.column_names();
	const std::string ipv4_name =
	    Squey::PVDuckDBQuery::quote_identifier(names[size_t(ipv4_col) + 1]); // +1: rowid
	const std::string string_name =
	    Squey::PVDuckDBQuery::quote_identifier(names[size_t(string_col) + 1]);

	PVCore::PVSelBitField out(row_count);

	// --- ipv4 keeps pvcop's ordering -----------------------------------------
	{
		// Read the column back through pvcop and split it on its median address,
		// so the expected count comes from pvcop rather than from the query.
		const pvcop::db::array& column = nraw.column(ipv4_col);
		std::vector<std::string> texts;
		texts.reserve(row_count);
		for (size_t row = 0; row < row_count; ++row) {
			texts.emplace_back(column.at(row));
		}

		// The address exposed to SQL is the stored uint32, so the threshold is
		// expressed the same way on both sides.
		const std::string sql =
		    "SELECT rowid FROM layers WHERE " + ipv4_name + " < 2130706433"; // 127.0.0.1
		query.select(sql, out);

		// pvcop-side oracle: rebuild the integer from the dotted form.
		size_t expected = 0;
		for (const std::string& text : texts) {
			unsigned a = 0, b = 0, c = 0, d = 0;
			if (std::sscanf(text.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
				const uint32_t value = (a << 24) | (b << 16) | (c << 8) | d;
				expected += static_cast<size_t>(value < 2130706433u);
			}
		}
		PV_VALID(count_selected(out, row_count), expected);
	}

	// A range predicate must select a contiguous address block, which a
	// lexicographic ordering would not.
	{
		query.select("SELECT rowid FROM layers WHERE " + ipv4_name +
		                 " BETWEEN 167772160 AND 184549375", // 10.0.0.0 -> 10.255.255.255
		             out);
		const pvcop::db::array& column = nraw.column(ipv4_col);
		for (size_t row = 0; row < row_count; ++row) {
			const bool selected = out.get_line(PVRow(row));
			const bool starts_with_10 = column.at(row).rfind("10.", 0) == 0;
			PV_ASSERT_VALID(selected == starts_with_10, "row", row);
		}
	}

	// --- string columns fall back to their textual form -----------------------
	{
		const pvcop::db::array& column = nraw.column(string_col);
		const std::string needle = column.at(0);

		query.select("SELECT rowid FROM layers WHERE " + string_name + " = '" + needle + "'",
		             out);

		size_t expected = 0;
		for (size_t row = 0; row < row_count; ++row) {
			expected += static_cast<size_t>(column.at(row) == needle);
		}
		PV_VALID(count_selected(out, row_count), expected);
		PV_ASSERT_VALID(out.get_line(PVRow(0)), "row", size_t(0));
	}

	return 0;
}
