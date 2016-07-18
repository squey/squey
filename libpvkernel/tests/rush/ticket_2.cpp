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
	pvtest::TestEnv env(TEST_FOLDER "/pvkernel/rush/tickets/2/apache.access",
	                    TEST_FOLDER "/pvkernel/rush/tickets/2/apache.access.format");

	env.load_data();

	PV_VALID(env.get_nraw_size(), 100000UL);

	return 0;
}
