/**
 * \file nraw_create.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>
#include <vector>

#include <tbb/tick_count.h>

#include <pvkernel/rush/PVNrawDiskBackend.h>
#include <pvkernel/core/picviz_bench.h>

#define N 2000000000

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	const char* nraw_path = argv[1];
	PVRush::PVNrawDiskBackend backend;

	backend.init(nraw_path, 1);
	backend.set_direct_mode(false);

	for (int i = 0 ; i < N; i++) {
		std::stringstream st;
		st << i << " ";
		backend.add(0, st.str().c_str(), st.str().length());
	}
	backend.flush();
}

