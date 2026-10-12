//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// The text the table models give their views for a long value.
//
// Qt lays out the whole text of a cell to elide it: a value of millions of
// characters took seconds at each repaint, and froze the listing. A table model gives
// at most DISPLAYED_SIZE_MAX bytes of a value, cut between two characters and
// followed by an ellipsis, in the cell as in its tooltip.

#include <pvkernel/core/squey_assert.h>

#include <squey/PVDuckDBQuery.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVSimpleStringListModel.h>
#include <pvguiqt/PVSqlResultModel.h>
#include <pvguiqt/PVStatsModel.h>

#include <pvcop/db/algo.h>
#include <pvcop/db/array.h>

#include <QApplication>
#include <QTemporaryDir>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "test-env.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="value" type="string">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr size_t MAX = PVGuiQt::PVAbstractTableModel::DISPLAYED_SIZE_MAX;
constexpr QChar ELLIPSIS(0x2026);

// Four bytes in UTF-8: after 0 to 3 ASCII characters, the cut falls on each of its
// bytes in turn.
constexpr const char* EMOJI = "\xF0\x9F\x98\x80";

struct Value {
	std::string imported;
	QString displayed;
};

std::vector<Value> expected_values()
{
	std::vector<Value> values{
	    {"short value", "short value"},
	    {std::string(MAX, 'a'), QString(MAX, u'a')},
	    {std::string(MAX + 1, 'b'), QString(MAX, u'b') + ELLIPSIS},
	    {std::string(size_t(1) << 23, 'c'), QString(MAX, u'c') + ELLIPSIS},
	};
	for (size_t lead = 0; lead < 4; ++lead) {
		std::string imported(lead, 'd');
		for (size_t i = 0; i <= MAX / 4; ++i) {
			imported += EMOJI;
		}
		values.push_back(
		    {imported,
		     QString(lead, u'd') + QString::fromUtf8(EMOJI).repeated((MAX - lead) / 4) + ELLIPSIS});
	}
	return values;
}

// The tooltip is checked when the model gives one.
void check(const char* name,
           const PVGuiQt::PVAbstractTableModel& model,
           const std::vector<Value>& expected,
           bool tooltip)
{
	PV_ASSERT_VALID(model.rowCount() == int(expected.size()), "model", name, "rows listed",
	                model.rowCount());

	for (int row = 0; row < int(expected.size()); ++row) {
		const QModelIndex index = model.index(row, 0);
		const QString displayed = model.data(index, Qt::DisplayRole).toString();
		std::cout << name << ", row " << row << ": " << expected[row].imported.size() << " bytes, "
		          << displayed.size() << " UTF-16 code units displayed" << std::endl;
		PV_ASSERT_VALID(displayed == expected[row].displayed, "model", name, "row", row,
		                "expected code units", expected[row].displayed.size());

		if (tooltip) {
			const QString text = model.data(index, Qt::ToolTipRole).toString();
			PV_ASSERT_VALID(QString(text).remove("<br>") == expected[row].displayed, "model", name,
			                "row", row, "tooltip code units", text.size());
		}
	}
}

} // namespace

int main(int argc, char** argv)
{
	init_env();

	const std::vector<Value> expected = expected_values();

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const QString csv = dir.filePath("values.csv");
	const QString format = dir.filePath("values.csv.format");
	{
		std::ofstream out(csv.toStdString(), std::ios::binary);
		// A byte order mark settles the charset, otherwise guessed from the first chunk:
		// filled with the 8 MiB of ASCII, it passed for ISO-8859-9.
		out << "\xEF\xBB\xBF";
		for (const Value& value : expected) {
			out << value.imported << "\n";
		}
		std::ofstream(format.toStdString()) << FORMAT;
	}

	auto root = std::make_unique<Squey::PVRoot>();
	Squey::PVSource& src = get_src_from_file(*root, csv, format);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View
	Squey::PVView& view = *src.current_view();

	QApplication app(argc, argv);

	check("listing", PVGuiQt::PVListingModel(view), expected, true);

	std::vector<std::string> imported;
	for (const Value& value : expected) {
		imported.push_back(value.imported);
	}

	// Each value counted once, as the distinct values of a column.
	pvcop::db::array counts("number_uint64", expected.size());
	auto& count = counts.to_core_array<uint64_t>();
	std::fill(count.begin(), count.end(), 1);
	pvcop::db::array total("number_uint64", 1);
	total.to_core_array<uint64_t>()[0] = expected.size();
	pvcop::db::array minmax = pvcop::db::algo::minmax(counts);
	check("stats",
	      PVGuiQt::PVStatsModel("Count", "value", QString(), pvcop::db::make_string_array(imported),
	                            std::move(counts), std::move(total), std::move(minmax)),
	      expected, true);

	Squey::PVDuckDBQuery::Table table;
	table.column_names = {"value"};
	table.column_types = {"VARCHAR"};
	for (const std::string& value : imported) {
		table.rows.push_back({value});
	}
	check("SQL result", PVGuiQt::PVSqlResultModel(std::move(table)), expected, false);

	PVGuiQt::PVSimpleStringListModel::container_type lines;
	for (size_t i = 0; i < imported.size(); ++i) {
		lines.emplace(i, imported[i]);
	}
	check("string list", PVGuiQt::PVSimpleStringListModel(lines), expected, false);

	return 0;
}
