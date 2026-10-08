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

// A date to the microsecond is an instant in SQL, not the text it was written as.
//
// A datetime_us column stores a boost ptime per row, and SQL used to see it as its
// text: a query could neither subtract two dates, nor truncate one, nor sort them
// otherwise than as strings. The scan now hands DuckDB the TIMESTAMP the ptime
// converts into, microseconds included. The file is written the way the FUSION
// furnace log writes its dates, day first, with a row whose date is missing.

#include "common.h"

#include <squey/PVDuckDBQuery.h>
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <QTemporaryDir>

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
   <axis name="when" type="time" type_format="dd.MM.yyyy H:m:s.SSSSSS">
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

// A second apart, then the last microsecond of a day, a row with no date, and the
// first microsecond of the next day.
const std::vector<std::string> WHEN = {"12.04.2015 07:13:30.890000", "12.04.2015 07:13:31.890000",
                                       "12.04.2015 23:59:59.999999", "",
                                       "13.04.2015 00:00:00.000001"};
// The same instants as DuckDB writes them.
const std::vector<std::string> INSTANTS = {"2015-04-12 07:13:30.89", "2015-04-12 07:13:31.89",
                                           "2015-04-12 23:59:59.999999", "",
                                           "2015-04-13 00:00:00.000001"};

Squey::PVView& first_view(pvtest::TestEnv& env)
{
	const auto& scenes = env.root.get_children();
	const auto& sources = (*scenes.begin())->get_children();
	return *(*sources.begin())->current_view();
}

//! One column of a query's answer, row by row -- a NULL as an empty string -- or the
//! error it raised.
std::vector<std::string> column_of(const Squey::PVDuckDBQuery& query, const std::string& sql,
                                   const PVCore::PVSelBitField* in = nullptr)
{
	std::vector<std::string> values;
	try {
		for (const auto& row : query.run_tabular(sql, in, 100).rows) {
			values.push_back(row.at(0));
		}
	} catch (const std::exception& e) {
		values = {std::string("error: ") + e.what()};
	}
	return values;
}

void expect(const Squey::PVDuckDBQuery& query, const std::string& sql,
            const std::vector<std::string>& wanted, const PVCore::PVSelBitField* in = nullptr)
{
	const std::vector<std::string> got = column_of(query, sql, in);
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
	const std::string csv = dir.filePath("furnace.csv").toStdString();
	const std::string format = dir.filePath("furnace.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (size_t i = 0; i < WHEN.size(); ++i) {
			out << WHEN[i] << "," << 100 * (i + 1) << "\n";
		}
		std::ofstream(format) << FORMAT;
	}
	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = first_view(env);
	PV_ASSERT_VALID(view.get_rushnraw_parent().column(PVCol(0)).formatter()->name() ==
	                    std::string("datetime_us"),
	                "the date axis", "is not a datetime_us one");
	const Squey::PVDuckDBQuery query(view);

	// An instant, to the microsecond; the missing date is NULL.
	expect(query, "SELECT typeof(\"when\") FROM layers LIMIT 1", {"TIMESTAMP"});
	expect(query, "SELECT \"when\" FROM layers ORDER BY rowid", INSTANTS);
	expect(query,
	       "SELECT strftime(\"when\", '%d.%m.%Y %H:%M:%S.%f') FROM layers WHERE \"when\" IS NOT "
	       "NULL ORDER BY rowid",
	       {WHEN[0], WHEN[1], WHEN[2], WHEN[4]});
	expect(query, "SELECT count(\"when\") FROM layers", {"4"});

	// Subtracted: a second, and two microseconds across midnight.
	expect(query,
	       "SELECT epoch_us(\"when\") - epoch_us(lag(\"when\") OVER (ORDER BY rowid)) FROM layers "
	       "WHERE \"when\" IS NOT NULL ORDER BY rowid",
	       {"", "1000000", "60388109999", "2"});
	// Truncated to the day, and sorted as instants.
	expect(query,
	       "SELECT count(*) FROM layers WHERE \"when\" IS NOT NULL GROUP BY date_trunc('day', "
	       "\"when\") ORDER BY 1",
	       {"1", "3"});
	expect(query, "SELECT power FROM layers WHERE \"when\" IS NOT NULL ORDER BY \"when\" DESC",
	       {"500", "300", "200", "100"});

	// Filtered by DuckDB, equality included: pvcop would read a TIMESTAMP literal
	// through the column's own pattern.
	expect(query, "SELECT power FROM layers WHERE \"when\" = TIMESTAMP '2015-04-12 07:13:31.89'",
	       {"200"});
	expect(query,
	       "SELECT power FROM layers WHERE \"when\" IN (TIMESTAMP '2015-04-12 07:13:30.89', "
	       "TIMESTAMP '2015-04-13 00:00:00.000001') ORDER BY rowid",
	       {"100", "500"});
	expect(query,
	       "SELECT power FROM layers WHERE \"when\" >= TIMESTAMP '2015-04-12 23:59:59.999999' "
	       "ORDER BY rowid",
	       {"300", "500"});

	// Rows picked out of order go through the scan's other path.
	Squey::PVSelection in(view.get_row_count());
	in.select_none();
	for (const PVRow row : {PVRow(1), PVRow(3), PVRow(4)}) {
		in.set_line(row, true);
	}
	expect(query, "SELECT \"when\" FROM selection ORDER BY rowid", {INSTANTS[1], "", INSTANTS[4]},
	       &in);

	// The text a date was written as stays within reach.
	expect(query, "SELECT \"when\" FROM selection(text := true) ORDER BY rowid LIMIT 1", {WHEN[0]});

	std::cout << "datetime_us: an instant to the microsecond, NULL where missing" << std::endl;
	return 0;
}
