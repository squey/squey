/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "common.h"

#include <pvkernel/core/inendi_assert.h>

#include <pvcop/db/algo.h>

int main()
{
	std::vector<std::string> inputs{TEST_FOLDER "/picviz/inc_1M.csv",
	                                TEST_FOLDER "/picviz/inc_1M.csv"};

	pvtest::TestEnv env(inputs, TEST_FOLDER "/formats/inc.csv.with_header.format", 1);

	const Inendi::PVSource* src = env.root.get_children<Inendi::PVSource>().front();
	PV_VALID(src->get_row_count(), 1999986U);

	size_t sum = (size_t)pvcop::db::algo::sum(src->get_rushnraw().column(PVCol(0)));
	PV_VALID(sum, 1000000999944U);

	return 0;
}
