//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

static constexpr const char* csv_win1 = TEST_FOLDER "/sources/windows_endings1.csv";
static constexpr const char* csv_win2 = TEST_FOLDER "/sources/windows_endings2.csv";
static constexpr const char* csv_win3 = TEST_FOLDER "/sources/windows_endings3.csv";
static constexpr const char* csv_win_format = TEST_FOLDER "/formats/windows_endings.format";
static constexpr const char* exported_win_csv_file =
#if __APPLE__
    TEST_FOLDER "/exports/multi_inputs_windows_endings_macos.csv";
#elif _WIN32
	TEST_FOLDER "/exports/multi_inputs_windows_endings_win.csv";
#else
    TEST_FOLDER "/exports/multi_inputs_windows_endings.csv";
#endif

static constexpr const char* csv_empty_last_col1 = TEST_FOLDER "/sources/empty_last_column1.csv";
static constexpr const char* csv_empty_last_col2 = TEST_FOLDER "/sources/empty_last_column2.csv";
static constexpr const char* csv_empty_last_col_format =
    TEST_FOLDER "/formats/empty_last_column.format";
static constexpr const char* exported_empty_last_col_csv_file =
    TEST_FOLDER "/exports/multi_inputs_empty_last_column.csv";

int main()
{
	pvtest::TestEnv env(std::vector<std::string>{csv_win1, csv_win2, csv_win3}, csv_win_format);
	env.add_source(std::vector<std::string>{csv_empty_last_col1, csv_empty_last_col2},
	               csv_empty_last_col_format);

	const auto& sources = env.root.get_children<Squey::PVSource>();
	PV_VALID(sources.size(), (size_t)2);

	auto exports =
	    std::vector<std::string>{exported_win_csv_file, exported_empty_last_col_csv_file};
	for (size_t i = 0; i < exports.size(); i++) {
		std::string out_path = pvtest::get_tmp_filename();
		// Dump the NRAW to file and check value is the same
		auto source_it = sources.begin();
		std::advance(source_it, i);
		const PVRush::PVNraw& nraw = (*source_it)->get_rushnraw();
		nraw.dump_csv(out_path);

		pvlogger::info() << exports[i] << " - " << out_path << std::endl;
		bool same_content = PVRush::PVUtils::files_have_same_content(exports[i], out_path);
		if (not same_content) {
		    pvlogger::error() << std::ifstream(out_path).rdbuf() << std::endl;
		}

		std::remove(out_path.c_str());
		PV_ASSERT_VALID(same_content);
	}

	return 0;
}
