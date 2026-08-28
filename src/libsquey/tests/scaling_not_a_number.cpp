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

// A NaN in a floating point column -- how numpy and pandas write a missing
// value, and what any export from them can carry -- cannot be compared, so it
// cannot be positioned along an axis. Scaling used to hand it to the arithmetic
// anyway and convert the resulting NaN to an unsigned integer, which is
// undefined: an assertion in a debug build, and a position at the far end of the
// axis in a release one, where it reads as a real value.
//
// It now goes where the axis keeps what it cannot place, the range reserved for
// invalid values, which is also where the import sends a NaN it did call
// invalid. The two verdicts had to agree: bounds computed over NaNs differ
// between compilers, so the same column reached the scaling with unordered
// bounds under gcc and ordered ones under clang.
//
// Infinities are left alone on purpose: they still compare, so they still order.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <squey/PVScaled.h>

#include "common.h"

#include <fstream>
#include <limits>
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
			// Both rows land where the axis keeps what it cannot place, which is
			// the position the invalid values are given. Asserting the position
			// rather than only the agreement is what tells the fix from the
			// undefined conversion it replaced: that one used to land here too,
			// but only by accident of how a NaN converts on this machine.
			PV_VALID(values[0], std::numeric_limits<uint32_t>::max(), "spelling", spelling,
			         "column", (size_t)col);
			PV_VALID(values[1], values[0], "spelling", spelling, "column", (size_t)col);
		}
	}

	return 0;
}
