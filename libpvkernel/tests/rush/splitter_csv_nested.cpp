/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <chrono>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

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
	env._ext.get_nraw().dump_csv(out_path);

	bool same_content = PVRush::PVUtils::files_have_same_content(out_path, ref_file);
	if (not same_content) {
		pvlogger::info() << out_path << std::endl;
	}
	PV_ASSERT_VALID(same_content);

	std::remove(out_path.c_str());

	return 0;
}
