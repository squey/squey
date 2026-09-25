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

// A script reads a column by the name the column has, whatever the view shows.
//
// Names were looked up among the axes on screen, which the interface hides and
// reorders: a hidden axis put its column out of reach of its own name while its
// index still read it.

#include "common.h"

#include <squey/PVPythonInterpreter.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include <QTemporaryDir>

#include <fstream>
#include <iostream>
#include <string>

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

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& view = *env.root.get_children<Squey::PVView>().front();
	// The last column first, the second one hidden.
	view.set_axes_combination({PVCol(2), PVCol(0)});

	try {
		Squey::PVPythonInterpreter::get(env.root).execute_script(R"(
source = squey.source(0)
assert list(source.column("a")) == [1, 2, 3, 4], list(source.column("a"))
assert list(source.column("b")) == [10, 20, 30, 40], list(source.column("b"))
assert list(source.column("c")) == [100, 200, 300, 400], list(source.column("c"))
assert source.column_type("b") == source.column_type(1), source.column_type("b")
)",
		                                                         false);
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		PV_ASSERT_VALID(false, "the script", "failed");
	}

	std::cout << "columns read by their own names, whatever the axes show" << std::endl;
	return 0;
}
