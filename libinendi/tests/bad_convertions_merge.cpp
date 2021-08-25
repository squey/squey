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

#include <inendi/PVSource.h>

#include <pvkernel/core/inendi_assert.h>

#include "common.h"

constexpr static size_t ROW_COUNT = 100000;

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/bad_convertions_merge.csv";
static constexpr const char* log_format =
    TEST_FOLDER "/pvkernel/rush/bad_convertions_merge.csv.format";

int main()
{
	pvtest::TestEnv env(log_file, log_format, 1);

	Inendi::PVSource& source = *env.root.get_children<Inendi::PVSource>().front();

	for (size_t i = 0; i < ROW_COUNT; i++) {
		if (i < ROW_COUNT / 2) {
			PV_VALID(source.get_value(i, PVCol(0)), std::string("test"));
		} else {
			PV_VALID(source.get_value(i, PVCol(0)), std::string("0.0.0.0"));
		}
		PV_VALID(source.get_value(i, PVCol(1)), std::string(""));
	};

	return 0;
}
