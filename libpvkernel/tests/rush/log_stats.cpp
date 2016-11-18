/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "../../tools/rush/log_stats.h"
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVUtils.h>

static constexpr const char* intput1 = TEST_FOLDER "/sources/proxy_sample1.log";
static constexpr const char* intput2 = TEST_FOLDER "/sources/proxy_sample2.log";
static constexpr const char* fileformat = TEST_FOLDER "/formats/proxy_sample.log.format";
static constexpr const char* ref_output = TEST_FOLDER "/picviz/log_stats_ref_output";

int main()
{
	pvtest::init_ctxt();

	std::string tmp_output = pvtest::get_tmp_filename();
	std::ofstream out(tmp_output);
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
