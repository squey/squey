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

#include <chrono>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include "helpers.h"
#include "common.h"

static constexpr const char* log_file =
    TEST_FOLDER "/pvkernel/rush/splitters/csv/splitter_csv_nested.csv";
static constexpr const char* log_format =
    TEST_FOLDER "/pvkernel/rush/splitters/csv/splitter_csv_nested.csv.format";
static constexpr const char* ref_file =
    TEST_FOLDER "/pvkernel/rush/splitters/csv/splitter_csv_nested.csv.out";

int main()
{
	pvtest::TestEnv env(log_file, log_format);

	env.load_data();

	std::string out_path = pvtest::get_tmp_filename();
	env._nraw.dump_csv(out_path);

	bool same_content = PVRush::PVUtils::files_have_same_content(out_path, ref_file);
	if (not same_content) {
		pvlogger::info() << out_path << std::endl;
	}
	PV_ASSERT_VALID(same_content);

	std::remove(out_path.c_str());

	return 0;
}
