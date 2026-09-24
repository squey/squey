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

// Opening a series view along a date read through ICU leaves the column reading
// as it did.
//
// The date range of the view edits a copy of the column's ends, and the copy
// shared the column's formatter: setting the range's own pattern on it rewrote how
// the whole column reads, and ICU, handed a strftime pattern, wrote garbage in the
// listing and everywhere else for the rest of the session.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVSeriesViewWidget.h>

#include <squey/PVView.h>

#include <QApplication>
#include <QTemporaryDir>

#include <fstream>
#include <iostream>
#include <string>

#include "common.h"

namespace
{

// A named zone is what only ICU reads, which makes this a datetime_ms column.
constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="when" type="time" type_format="yyyy-MM-dd HH:mm:ss zzz">
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

} // namespace

int main(int argc, char** argv)
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("furnace.csv").toStdString();
	const std::string format = dir.filePath("furnace.csv.format").toStdString();
	{
		std::ofstream out(csv);
		out << "2015-04-12 07:13:30 GMT,100\n"
		    << "2015-04-12 08:13:30 GMT,200\n"
		    << "2015-04-12 09:13:30 GMT,300\n";
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	const pvcop::db::array& when = view->get_rushnraw_parent().column(PVCol(0));
	PV_ASSERT_VALID(when.formatter()->name() == std::string("datetime_ms"), "the date column",
	                "is not read through ICU", "formatter", when.formatter()->name());
	const std::string before = when.at(0);

	// After TestEnv, which runs an application of its own while it builds.
	if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
		qputenv("QT_QPA_PLATFORM", "offscreen");
	}
	QApplication app(argc, argv);

	// The abscissa selector asks the display registry whether the series view takes
	// a given axis, and the registry is filled when the backend comes up.
	PVParallelView::common::RAII_backend_init backend_resources;

	{
		PVParallelView::PVSeriesViewWidget widget(view, PVCol(0));
		const std::string after = when.at(0);
		PV_ASSERT_VALID(after == before, "the date column read", before, "and reads now", after);
	}

	std::cout << "a series view leaves the column's dates as they read: " << before << std::endl;
	return 0;
}
