/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

int main()
{
	pvtest::TestEnv env(TEST_FOLDER "/sources/proxy.log",
	                    TEST_FOLDER "/formats/proxy.log.with_header.format", 1);

	PV_VALID(env.root.get_children<Inendi::PVSource>().front()->get_row_count(), 99997U);

	return 0;
}
