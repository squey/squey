/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <fstream>

#include <pvkernel/core/inendi_assert.h>

#include "common.h"

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/splitters/mac/mac";
const std::string fileformat = TEST_FOLDER "/picviz/string_default_mapping.csv.format";

/**
 * Load data in the NRaw.
 */
int main()
{
	pvtest::TestEnv env(log_file, fileformat);

	env.load_data();

	return 0;
}
