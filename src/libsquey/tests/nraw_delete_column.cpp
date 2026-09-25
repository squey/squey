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

// Removing a column shifts every column after it down by one, and each still reads
// as itself.
//
// The nraw keeps an array per column, which delete_column() used to leave as it
// was: the collection had one column fewer and the arrays as many as before, so
// every column past the one removed read as its neighbour. A column appended after
// the import comes last, where only a lined-up vector reaches it.

#include "common.h"

#include <squey/PVSource.h>

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <QTemporaryDir>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="a" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="b" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="c" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

} // namespace

int main()
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("values.csv").toStdString();
	const std::string format = dir.filePath("values.csv.format").toStdString();
	{
		std::ofstream(csv) << "1,10,100\n2,20,200\n3,30,300\n4,40,400\n";
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::Source);
	PVRush::PVNraw& nraw = env.root.get_children<Squey::PVSource>().front()->get_rushnraw();

	const std::vector<uint32_t> appended = {1000, 2000, 3000, 4000};
	PV_ASSERT_VALID(nraw.append_column("number_uint32", std::as_bytes(std::span(appended))),
	                "a column", "could not be appended");

	nraw.delete_column(PVCol(0));
	PV_ASSERT_VALID(nraw.column_count() == PVCol(3), "columns left", nraw.column_count().value());

	const std::vector<std::vector<std::string>> expected = {
	    {"10", "20", "30", "40"}, {"100", "200", "300", "400"}, {"1000", "2000", "3000", "4000"}};
	for (PVCol col(0); col < nraw.column_count(); col++) {
		for (PVRow row = 0; row < nraw.row_count(); row++) {
			const std::string& wanted = expected[col.value()][row];
			PV_ASSERT_VALID(nraw.at_string(row, col) == wanted, "column", col.value(), "row", row,
			                "reads", nraw.at_string(row, col), "instead of", wanted);
		}
	}

	std::cout << "after removing the first column, every other one reads as itself" << std::endl;
	return 0;
}
