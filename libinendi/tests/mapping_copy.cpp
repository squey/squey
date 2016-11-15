/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <inendi/PVMapped.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvcop/db/algo.h>

#include "common.h"

#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
// 10 000 000 lines.
static constexpr int dupl = 200;
#else
static constexpr int dupl = 1;
#endif

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " input_file format" << std::endl;
		return 1;
	}

	const char* input_file = argv[1];
	const char* format = argv[2];

	pvtest::TestEnv env(input_file, format, dupl);

	auto start = std::chrono::system_clock::now();

	Inendi::PVMapped& mapped = env.compute_mapping();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	PVRush::PVNraw const& nraw = env.root.get_children<Inendi::PVSource>().front()->get_rushnraw();

	// Check mapping is the same as NRaw values
	PV_ASSERT_VALID(mapped.get_column(0) == nraw.column(0));
#else
	(void)mapped;
#endif

	return 0;
}
