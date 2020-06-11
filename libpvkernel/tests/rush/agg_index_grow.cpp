/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

int main()
{
	pvtest::TestEnv env(TEST_FOLDER "/pvkernel/rush/long_line.csv",
	                    TEST_FOLDER "/pvkernel/rush/formats/tiny.csv.format");

	env.load_data();

	PV_VALID(env.get_nraw_size(), 3UL);

	PV_VALID(env._nraw.get_valid_row_count(), 3UL);
	PV_VALID(env._nraw.valid_rows_sel().get_line(2), true);

	return 0;
}
