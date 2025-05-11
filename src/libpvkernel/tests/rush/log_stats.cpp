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

#include "../../tools/rush/log_stats.h"
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVUtils.h>

static constexpr const char* intput1 = TEST_FOLDER "/sources/proxy_sample1.log";
static constexpr const char* intput2 = TEST_FOLDER "/sources/proxy_sample2.log";
static constexpr const char* fileformat = TEST_FOLDER "/formats/proxy_sample.log.format";
static constexpr const char* ref_output = TEST_FOLDER "/picviz/log_stats_ref_output";

int main()
{
	pvtest::init_ctxt();

	std::string tmp_output = pvtest::get_tmp_filename();
	std::ofstream out{std::filesystem::path(tmp_output)};
	std::streambuf* coutbuf = std::cout.rdbuf();
	std::cout.rdbuf(out.rdbuf());

	run_stats({
	    {intput1, intput2}, // inputs
	    fileformat,         // format
	    {{2, 4, 5}},        // columns
	    true,               // extended stats
	    7                   // max stats rows
	});

	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(tmp_output, ref_output));

	std::cout.rdbuf(coutbuf);
}
