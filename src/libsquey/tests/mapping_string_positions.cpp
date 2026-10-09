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

// Where the string mapping puts strings on their axis.

#include <squey/PVMapped.h>

#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <QTemporaryDir>

#include <fstream>
#include <string>
#include <vector>

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="text" type="string">
    <mapping mode="string"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

} // namespace

int main()
{
	std::vector<std::string> rows;
	auto row = [&rows](std::string text) {
		rows.push_back(std::move(text));
		return rows.size() - 1;
	};

	// Past the first byte, strings of a length are told apart by the sum of the
	// following ones, however long they are.
	const std::string url = "https://www.example.com/" + std::string(126, 'a');
	std::string changed = url;
	changed[100] = 'b';
	const size_t long_url = row(url);
	const size_t changed_url = row(changed);

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("strings.csv").toStdString();
	const std::string format = dir.filePath("strings.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (std::string const& text : rows) {
			out << text << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format);
	auto const& position = env.compute_mapping().get_column(PVCol(0)).to_core_array<uint32_t>();

	PV_ASSERT_VALID(position[long_url] != position[changed_url], "why",
	                "the bytes past the first tell long strings of a length apart");

	return 0;
}
