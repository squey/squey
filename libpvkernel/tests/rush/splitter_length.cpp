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

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include "common.h"

#include <iostream>

static constexpr const char* log_file = "/tmp/test-splitter-length.input";
static constexpr const char* ref_file = "/tmp/test-splitter-length.ref";

struct testcase_t {
	int length;
	bool from_left;
	std::vector<std::string> result;
};

static const char* test_text = "abcdefghijklmnopqrstuvwxyz";

static const testcase_t testcases[] = {{-42, true, {"", "abcdefghijklmnopqrstuvwxyz"}},
                                       {-42, false, {"abcdefghijklmnopqrstuvwxyz", ""}},
                                       {0, true, {"", "abcdefghijklmnopqrstuvwxyz"}},
                                       {0, false, {"abcdefghijklmnopqrstuvwxyz", ""}},
                                       {5, true, {"abcde", "fghijklmnopqrstuvwxyz"}},
                                       {5, false, {"abcdefghijklmnopqrstu", "vwxyz"}},
                                       {26, true, {"abcdefghijklmnopqrstuvwxyz", ""}},
                                       {26, false, {"", "abcdefghijklmnopqrstuvwxyz"}},
                                       {42, true, {"abcdefghijklmnopqrstuvwxyz", ""}},
                                       {42, false, {"", "abcdefghijklmnopqrstuvwxyz"}}};

int main()
{
	pvtest::TestSplitter ts;

	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("length");

	auto ff =
	    std::unique_ptr<PVFilter::PVElementFilterByFields>(new PVFilter::PVElementFilterByFields());
	ff->add_filter(sp_lib_p);
	PVFilter::PVChunkFilterByElt chk_flt{std::move(ff)};

	std::ofstream of(log_file);
	of << test_text << std::endl;
	of.close();

	for (const auto& testcase : testcases) {
		ts.reset(log_file);

		/* initializing the splitter
		 */
		PVCore::PVArgumentList args = sp_lib_p->default_args();
		args["length"] = testcase.length;
		args["from_left"] = testcase.from_left;

		sp_lib_p->set_args(args);

		/* initializing the reference file
		 */
		std::ofstream of(ref_file);
		of << "'" << testcase.result[0] << "','" << testcase.result[1] << "'" << std::endl;
		of.close();

		/* let's go
		 */
		auto res = ts.run_normalization(chk_flt);
		size_t nelts_org = std::get<0>(res);
		size_t nelts_valid = std::get<1>(res);
		std::string output_file = std::get<2>(res);

		PV_VALID(nelts_valid, nelts_org);

		/* checking output
		 */
		PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file), "ref_file",
		                ref_file, "output_file", output_file);

		std::remove(output_file.c_str());
	}

	std::remove(log_file);
	std::remove(ref_file);

	return 0;
}
