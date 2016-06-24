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
	pvtest::TestEnv env(TEST_FOLDER "/sources/proxy.log", TEST_FOLDER "/formats/proxy.log.format",
	                    1);

	PV_VALID(env.root.get_children<Inendi::PVSource>().front()->get_row_count(), 100000U);

	return 0;
}
