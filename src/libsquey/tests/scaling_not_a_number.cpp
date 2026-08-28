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

// A single "nan" in a floating point column -- how numpy and pandas write a
// missing value, and what any export from them can carry -- leaves the column's
// bounds unordered: neither ymin == ymax nor ymax > ymin holds. Scaling then
// computed a NaN ratio and converted it to an unsigned integer, which is
// undefined: an assertion in a debug build, and arbitrary positions along the
// axis in a release one, with nothing said to the user either way.
//
// Infinities are left alone on purpose: they still compare, so they still order.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <squey/PVScaled.h>

#include "common.h"

#include <fstream>
#include <string>

static std::string write_file(const std::string& path, const std::string& content)
{
	std::ofstream out(path);
	out << content;
	out.close();
	return path;
}

static const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="5" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis titlecolor="#ff921d" type="string" group="" tag="" color="#ffffff" name="name" mapping="default" key="false" plotting="default">
    <mapping mode="default"/>
    <plotting mode="default"/>
   </axis>
  </field>
  <field>
   <axis titlecolor="#ff921d" type="number_double" group="" tag="" color="#ffffff" name="value" mapping="default" key="false" plotting="default">
    <mapping mode="default"/>
    <plotting mode="default"/>
   </axis>
  </field>
  <field>
   <axis titlecolor="#ff921d" type="number_double" group="" tag="" color="#ffffff" name="logvalue" mapping="default" key="false" plotting="log">
    <mapping mode="default"/>
    <plotting mode="log"/>
   </axis>
  </field>
 </splitter>
</param>
)";

int main()
{
	const std::string dir = PVRush::PVNrawCacheManager::nraw_dir().toStdString();
	const std::string format = write_file(dir + "/nan_scaling.format", FORMAT);

	// A column whose values are all NaN is what leaves the bounds unordered:
	// where a NaN sits among ordinary values it is marked invalid and excluded
	// from them, and the scaling never sees it.
	for (const char* spelling : {"nan", "NaN", "-nan"}) {
		const std::string body = std::string("a,") + spelling + "," + spelling + "\nb," +
		                         spelling + "," + spelling + "\n";
		const std::string csv = write_file(dir + "/nan_scaling.csv", body);

		pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);

		Squey::PVSource& src = *env.root.get_children<Squey::PVSource>().front();
		PV_VALID((size_t)src.get_rushnraw().row_count(), size_t(2), "spelling", spelling);

		Squey::PVView* view = src.current_view();
		PV_ASSERT_VALID(view != nullptr, "spelling", spelling);

		// Both rows land at the same place. No order can be told between two
		// NaNs, and the one thing that must not happen is an undefined
		// conversion deciding it for us -- which is what the missing bound check
		// used to allow, after tripping an assertion in a debug build.
		// Both the linear and the logarithmic scaling are checked: they had the
		// same guard, written the same way, and it was wrong in both.
		const Squey::PVScaled& scaled = view->get_parent<Squey::PVScaled>();
		for (PVCol col : {PVCol(1), PVCol(2)}) {
			const uint32_t* values = scaled.get_column_pointer(col);
			PV_ASSERT_VALID(values != nullptr, "spelling", spelling, "column", (size_t)col);
			// Where exactly they land is not asserted: whether a NaN is taken for
			// an invalid value -- which earns the reserved position at the end of
			// the axis -- or for an ordinary one -- which lands mid-axis -- differs
			// between platforms and compilers. What must hold everywhere is that
			// the rows agree with each other, and that getting here at all took no
			// undefined conversion on the way.
			PV_VALID(values[1], values[0], "spelling", spelling, "column", (size_t)col);
		}
	}

	return 0;
}
