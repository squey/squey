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

// Differential ordering test for the business types.
//
// A wrong mapping for these does not fail loudly: the queries still return
// rows, just in the wrong order or matching the wrong ones. So the check is to
// order the same column two ways -- through SQL, and through a reference
// computed outside of both DuckDB and the mapping -- and require the two
// sequences to agree.
//
// For IPv6 the reference is inet_pton() plus memcmp() on the 16 network-order
// bytes, which is the definition of address order and owes nothing to how pvcop
// or DuckDB store the value. That is what makes the test able to catch a byte
// order mistake: pvcop keeps a __uint128_t in host order, DuckDB's uhugeint_t
// is {lower, upper}, and the two only line up on a little-endian host.
//
// Run as: <binary> <csv> <format> <column-type>

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvcop/db/array.h>

#include <arpa/inet.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "common.h"

/**
 * Order rows by IPv6 address, computed from the textual form so that the
 * expected order depends on neither pvcop's nor DuckDB's representation.
 */
static std::vector<size_t> reference_order_ipv6(const pvcop::db::array& column, size_t row_count)
{
	std::vector<std::array<unsigned char, 16>> keys(row_count);
	for (size_t row = 0; row < row_count; ++row) {
		const std::string text = column.at(row);
		in6_addr addr{};
		if (inet_pton(AF_INET6, text.c_str(), &addr) != 1) {
			// pvcop also accepts an IPv4 in an IPv6 column; mirror its mapping
			// onto the IPv4-in-IPv6 range so both sides agree.
			uint32_t v4 = 0;
			PV_ASSERT_VALID(inet_pton(AF_INET, text.c_str(), &v4) == 1, "row", row);
			std::memset(&addr, 0, sizeof(addr));
			std::memcpy(&addr.s6_addr[0], &v4, 4);
			addr.s6_addr[4] = 0xFF;
			addr.s6_addr[5] = 0xFF;
		}
		std::memcpy(keys[row].data(), addr.s6_addr, 16);
	}

	std::vector<size_t> order(row_count);
	std::iota(order.begin(), order.end(), size_t(0));
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		return std::memcmp(keys[a].data(), keys[b].data(), 16) < 0;
	});
	return order;
}

/**
 * Order rows by the stored integer, for the types whose storage is already the
 * sort key (epochs). The reference still comes from outside the query: it reads
 * the column through pvcop and sorts it here.
 */
static std::vector<size_t> reference_order_integer(const pvcop::db::array& column,
                                                   size_t row_count)
{
	const auto& values = column.to_core_array<uint64_t>();
	std::vector<size_t> order(row_count);
	std::iota(order.begin(), order.end(), size_t(0));
	std::stable_sort(order.begin(), order.end(),
	                 [&](size_t a, size_t b) { return values[a] < values[b]; });
	return order;
}

int main(int argc, char** argv)
{
	PV_ASSERT_VALID(argc == 4, "argc", argc);
	const std::string csv = argv[1];
	const std::string format = argv[2];
	const std::string wanted_type = argv[3];

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.current_view();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const size_t row_count = nraw.row_count();

	PVCol target(PVCol::value_type(-1));
	for (PVCol c(0); c < nraw.column_count(); ++c) {
		if (nraw.column(c).type() == wanted_type) {
			target = c;
			break;
		}
	}
	PV_ASSERT_VALID(target != PVCol::value_type(-1), "type", wanted_type);

	Squey::PVDuckDBQuery query(view->get_parent<Squey::PVSource>());
	const std::vector<std::string> names = query.column_names();
	const std::string column_name =
	    Squey::PVDuckDBQuery::quote_identifier(names[size_t(target) + 1]); // +1: rowid

	const pvcop::db::array& column = nraw.column(target);
	const std::vector<size_t> expected = (wanted_type == "ipv6")
	                                         ? reference_order_ipv6(column, row_count)
	                                         : reference_order_integer(column, row_count);

	// The SQL side. ORDER BY is what exercises the mapping: it compares values
	// the way the exposed type says they compare.
	std::vector<size_t> actual;
	actual.reserve(row_count);
	query.for_each_row("SELECT rowid FROM layers ORDER BY " + column_name + ", rowid",
	                   [&](size_t row) { actual.push_back(row); });

	PV_VALID(actual.size(), row_count);

	// Equal values may be ordered differently on each side, so compare the
	// values reached rather than the row ids themselves.
	for (size_t i = 0; i < row_count; ++i) {
		const std::string a = column.at(actual[i]);
		const std::string b = column.at(expected[i]);
		PV_ASSERT_VALID(a == b, "position", i, "sql", a, "reference", b);
	}

	return 0;
}
