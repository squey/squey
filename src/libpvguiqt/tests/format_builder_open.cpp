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

// The format builder opens the formats it saves.
//
// It saves one with fewer than two axes if the user insists, a format in progress for
// instance. Importing refuses such a format, and the builder is where it gets its axes:
// the builder has to open it. A PVRush::PVFormat refuses a format without any axis, by
// an exception that nothing catches before the event loop: opening one must not need
// it. A format with axes comes back with its combination of axes.

#include <pvkernel/core/squey_assert.h>

#include <pvguiqt/format-builder/PVXmlDomModel.h>

#include <QTemporaryDir>

#include <fstream>
#include <string>

#include "test-env.h"

namespace
{

// A splitter just added, before any field.
constexpr const char* SPLITTER_ONLY = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;"/>
</param>
)";

constexpr const char* ONE_AXIS = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="first" type="string"/>
  </field>
 </splitter>
</param>
)";

constexpr const char* COMBINED = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="first" type="string"/>
  </field>
  <field>
   <axis name="second" type="string"/>
  </field>
  <field>
   <axis name="third" type="string"/>
  </field>
 </splitter>
 <axes-combination>2,0</axes-combination>
</param>
)";

QString write(QTemporaryDir const& dir, const char* name, const char* content)
{
	const QString path = dir.filePath(name);
	std::ofstream(path.toStdString()) << content;
	return path;
}

void check_open(QString const& path, size_t axes, std::string const& combination)
{
	App::PVXmlDomModel model;
	PV_ASSERT_VALID(model.openXml(path), "format", path.toStdString());
	PV_VALID(model.get_axes_count(), axes, "format", path.toStdString());
	PV_VALID(model.get_axes_combination().to_string().toStdString(), combination, "format",
	         path.toStdString());
}

} // namespace

int main()
{
	init_env();

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");

	// What the builder writes for a new format, saved as it is.
	const QString empty = dir.filePath("empty.format");
	PV_ASSERT_VALID(App::PVXmlDomModel().saveXml(empty), "format", empty.toStdString());
	check_open(empty, 0, "");

	check_open(write(dir, "splitter_only.format", SPLITTER_ONLY), 0, "");
	check_open(write(dir, "one_axis.format", ONE_AXIS), 1, "0");
	check_open(write(dir, "combined.format", COMBINED), 3, "2,0");

	return 0;
}
