//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// A date a query reads is the date the listing shows, whatever the machine's zone.
//
// A time axis reaches SQL as its epoch, which a query turns back into an instant
// with to_timestamp(). That instant carries a zone, and everything a query then
// asks of it -- its day, its hour, its text, the day it falls in -- is the ICU
// extension's work: the build did not link it and a query cannot load it, so all
// of those failed. Linked, it reads instants in the machine's zone, while Squey
// reads a date written without one as UTC: a furnace's night shift would fall on
// the next day in Tokyo. The engine reads them in UTC, as the listing does.
//
// The zone is set to Tokyo's before the engine is made, which is where ICU looks it
// up: nine hours from UTC, so that a day, an hour and a text all move with it.

#include "common.h"

#include <squey/PVDuckDBQuery.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <QTemporaryDir>
#include <QtGlobal>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="when" type="time" type_format="yyyy-MM-dd HH:mm:ss">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="power" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

// Late in the evening, just after midnight, and the last second of a year.
const std::vector<std::string> WHEN = {"2020-03-01 23:30:00", "2020-03-02 00:15:00",
                                       "2020-12-31 23:59:59"};

Squey::PVView& first_view(pvtest::TestEnv& env)
{
	const auto& scenes = env.root.get_children();
	const auto& sources = (*scenes.begin())->get_children();
	return *(*sources.begin())->current_view();
}

//! One column of a query's answer, row by row; the error it raised if it raised one.
std::vector<std::string> column_of(const Squey::PVDuckDBQuery& query, const std::string& sql)
{
	std::vector<std::string> values;
	try {
		for (const auto& row : query.run_tabular(sql, nullptr, 100).rows) {
			values.push_back(row.at(0));
		}
	} catch (const std::exception& e) {
		values = {std::string("error: ") + e.what()};
	}
	return values;
}

void expect(const Squey::PVDuckDBQuery& query, const std::string& sql,
            const std::vector<std::string>& wanted)
{
	const std::vector<std::string> got = column_of(query, sql);
	std::string shown;
	for (const std::string& value : got) {
		shown += (shown.empty() ? "" : " | ") + value;
	}
	PV_ASSERT_VALID(got == wanted, "the query", sql, "answered", shown);
}

} // namespace

int main()
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("shifts.csv").toStdString();
	const std::string format = dir.filePath("shifts.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (size_t i = 0; i < WHEN.size(); ++i) {
			out << WHEN[i] << "," << 100 * (i + 1) << "\n";
		}
		std::ofstream(format) << FORMAT;
	}
	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = first_view(env);

	// What the listing shows, which is what the file says.
	const PVRush::PVNraw& nraw = view.get_rushnraw_parent();
	for (size_t row = 0; row < WHEN.size(); ++row) {
		PV_ASSERT_VALID(nraw.at_string(PVRow(row), PVCol(0)) == WHEN[row], "the listing shows",
		                nraw.at_string(PVRow(row), PVCol(0)));
	}

	// After the import, whose formatters set TZ to GMT for themselves.
	qputenv("TZ", "Asia/Tokyo");
	const Squey::PVDuckDBQuery query(view);

	expect(query, "SELECT current_setting('TimeZone')", {"UTC"});

	// The text of each instant is the listing's.
	expect(query,
	       "SELECT strftime(to_timestamp(\"when\"), '%Y-%m-%d %H:%M:%S') FROM layers ORDER BY rowid",
	       WHEN);
	// Its day: in Tokyo, all three would fall on the next one.
	expect(query, "SELECT CAST(to_timestamp(\"when\") AS DATE) FROM layers ORDER BY rowid",
	       {"2020-03-01", "2020-03-02", "2020-12-31"});
	// Its hour, and the day it is truncated to.
	expect(query, "SELECT hour(to_timestamp(\"when\")) FROM layers ORDER BY rowid",
	       {"23", "0", "23"});
	expect(query,
	       "SELECT strftime(date_trunc('day', to_timestamp(\"when\")), '%Y-%m-%d %H:%M') "
	       "FROM layers ORDER BY rowid",
	       {"2020-03-01 00:00", "2020-03-02 00:00", "2020-12-31 00:00"});
	// A count per day, which is how a query reads a rate: three days, where Tokyo
	// would put the first two instants on the same one.
	expect(query,
	       "SELECT count(*) FROM layers GROUP BY CAST(to_timestamp(\"when\") AS DATE) ORDER BY 1",
	       {"1", "1", "1"});
	// And an instant shown as it is, which says its zone.
	expect(query, "SELECT CAST(to_timestamp(0) AS VARCHAR)", {"1970-01-01 00:00:00+00"});

	// A query cannot move the zone: the configuration is locked.
	const std::vector<std::string> moved = column_of(query, "SET TimeZone = 'Asia/Tokyo'");
	PV_ASSERT_VALID(moved.size() == 1 and moved[0].rfind("error: ", 0) == 0,
	                "a query set the zone", moved.empty() ? std::string() : moved[0]);
	expect(query, "SELECT current_setting('TimeZone')", {"UTC"});

	std::cout << "instants read in UTC, as the listing shows them" << std::endl;
	return 0;
}
