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

// The values a CSV imports whose quoted values hold line feeds, and quotes, and the CSV
// Squey exports them to, which reads back the same.

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

std::string temporary_file(const char* name, const std::string& content)
{
	const QString path = PVCore::mkstemp(PVRush::PVNrawCacheManager::nraw_dir() + "/" + name);
	PV_ASSERT_VALID(not path.isEmpty(), "file", std::string(name));
	std::ofstream(path.toStdString(), std::ios::binary) << content;
	return path.toStdString();
}

} // namespace

int main()
{
	pvtest::init_ctxt();

	// each row, and the values it holds once imported
	const std::vector<std::pair<std::string, std::vector<std::string>>> rows{
	    {"1,\"first line\nsecond line\",10", {"1", "first line\nsecond line", "10"}},
	    {"2,plain,20", {"2", "plain", "20"}},
	    {"3,\"with \"\"quotes\"\"\r\nand a line feed\",30",
	     {"3", "with \"quotes\"\r\nand a line feed", "30"}},
	    {"4,\"C:\\dir\\\",40", {"4", "C:\\dir\\", "40"}},
	    {"5,\"say \\\"hi\\\"\",50", {"5", "say \"hi\"", "50"}},
	    {"6,\"a,b\",60", {"6", "a,b", "60"}},
	    {"7,\"\n\",70", {"7", "\n", "70"}},
	    {"8,12\" pipe,80", {"8", "12\" pipe", "80"}},
	    {"9,\"first line\nsecond line\",90", {"9", "first line\nsecond line", "90"}},
	};

	std::string csv;
	for (const auto& row : rows) {
		csv += row.first + "\n";
	}
	const std::string csv_path = temporary_file("csv_quoted_line_feeds_XXXXXX.csv", csv);

	std::string format = "<!DOCTYPE PVParamXml>\n<param version=\"" PVFORMAT_CURRENT_VERSION
	                     "\">\n<splitter type=\"csv\" sep=\",\" quote=\"&quot;\">\n";
	for (const char* name : {"id", "comment", "value"}) {
		format += std::string("<field><axis type=\"string\" name=\"") + name + "\"/></field>\n";
	}
	format += "</splitter>\n</param>\n";
	const std::string format_path = temporary_file("csv_quoted_line_feeds_XXXXXX.format", format);

	const auto check_import = [&rows, &format_path](const std::string& path) {
		pvtest::TestEnv env(path, format_path);
		env.load_data();

		PV_VALID(env.get_nraw_size(), rows.size(), "file", path);
		std::string exported;
		for (size_t r = 0; r < rows.size(); r++) {
			for (size_t c = 0; c < 3; c++) {
				PV_VALID(env._nraw.at_string(PVRow(r), PVCol(c)), rows[r].second[c], "file", path,
				         "row", r, "column", c);
			}
			exported +=
			    env._nraw.export_line(PVRow(r), {PVCol(0), PVCol(1), PVCol(2)}, ",", "\"") + "\n";
		}
		return exported;
	};

	const std::string exported = check_import(csv_path);
	const std::string exported_path =
	    temporary_file("csv_quoted_line_feeds_exported_XXXXXX.csv", exported);
	check_import(exported_path);

	std::remove(csv_path.c_str());
	std::remove(exported_path.c_str());
	std::remove(format_path.c_str());

	return 0;
}
