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

// A script working on a source whose view is not the current one.
//
// squey.source(N) went through the window's current view, which is another
// source's as soon as a second one has been opened after it, and null when none is
// current. It dereferenced it all the same: reading the first source's selection
// or column types, or adding a layer to it, crashed the application.

#include "common.h"

#include <squey/PVPythonInterpreter.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
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
		std::ofstream(csv) << "1,10\n2,20\n3,30\n4,40\n";
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView& first = *env.root.get_children<Squey::PVView>().front();

	// A second source, whose view becomes the current one as it is made.
	Squey::PVSource& second = env.add_source(csv, format);
	second.emplace_add_child().emplace_add_child().emplace_add_child();
	PV_ASSERT_VALID(env.root.current_view() != &first, "the first source's view",
	                "is still the current one");

	const int layers = first.get_layer_stack().get_layer_count();
	try {
		Squey::PVPythonInterpreter::get(env.root).execute_script(R"(
first = squey.source(0)
assert first.selection().size() == 4, first.selection().size()
assert first.column_type("b") != "", first.column_type("b")
first.insert_layer("from a script")
)",
		                                                         false);
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		PV_ASSERT_VALID(false, "the script", "failed");
	}
	PV_ASSERT_VALID(first.get_layer_stack().get_layer_count() == layers + 1, "layers of the first",
	                first.get_layer_stack().get_layer_count(), "were", layers);

	std::cout << "squey.source(0) works on its own view while another one is current"
	          << std::endl;
	return 0;
}
