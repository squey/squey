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
#include <utility>
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
  <field>
   <axis name="text in lower case" type="string">
    <mapping mode="string" convert-lowercase="true"/>
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

	// Past the longest length the mapping tells apart, strings stay above the
	// shorter ones instead of wrapping around.
	const size_t sixteen_kb = row("h" + std::string(16383, 'a'));
	std::vector<std::pair<size_t, size_t>> longer;
	for (size_t length : {32767, 32768, 65536, 65537, 100000}) {
		longer.emplace_back(length, row("h" + std::string(length - 1, 'a')));
	}

	// Asked to, the mapping places strings as if they were in lower case.
	const size_t capitalized = row("Squey");
	const size_t lower_case = row("squey");
	const size_t last_upper = row("squeY");

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const std::string csv = dir.filePath("strings.csv").toStdString();
	const std::string format = dir.filePath("strings.csv.format").toStdString();
	{
		std::ofstream out(csv);
		for (std::string const& text : rows) {
			out << text << "," << text << "\n";
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format);
	Squey::PVMapped& mapped = env.compute_mapping();
	auto const& position = mapped.get_column(PVCol(0)).to_core_array<uint32_t>();
	auto const& lowered_position = mapped.get_column(PVCol(1)).to_core_array<uint32_t>();

	PV_ASSERT_VALID(position[long_url] != position[changed_url], "why",
	                "the bytes past the first tell long strings of a length apart");
	for (auto const& [length, index] : longer) {
		PV_ASSERT_VALID(position[index] > position[sixteen_kb], "why",
		                "a longer string is placed above a shorter one", "length", length);
	}
	PV_ASSERT_VALID(position[capitalized] != position[lower_case] and
	                    position[last_upper] != position[lower_case],
	                "why", "case tells strings apart, unless asked otherwise");
	PV_ASSERT_VALID(lowered_position[capitalized] == lowered_position[lower_case] and
	                    lowered_position[last_upper] == lowered_position[lower_case],
	                "why", "in lower case, strings differing by case are the same");
	PV_ASSERT_VALID(lowered_position[lower_case] == position[lower_case], "why",
	                "a string already in lower case keeps its position");

	return 0;
}
