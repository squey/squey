/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

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
	// Prepare splitter plugin
	PVFilter::PVFieldsSplitter::p_type sp_lib_p =
	    LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_class_by_name("length");

	PVFilter::PVChunkFilterByElt chk_flt{std::unique_ptr<PVFilter::PVElementFilterByFields>(
	    new PVFilter::PVElementFilterByFields(sp_lib_p->f()))};

	std::ofstream of(log_file);
	of << test_text << std::endl;
	of.close();

	for (const auto& testcase : testcases) {
		pvtest::TestSplitter ts(log_file);

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
