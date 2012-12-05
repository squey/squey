/**
 * \file transp.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>

#include <tbb/tick_count.h>

#include <iostream>
#include <sstream>

#include <pvkernel/core/picviz_stat.h>

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " height width" << std::endl;
		return 0;
	}

	size_t nrows = atol(argv[1]);
	size_t ncols = atol(argv[2]);

	PVCore::PVMatrix<float, uint32_t, uint32_t> matrix,transp;
	matrix.resize(nrows, ncols, 0);
	transp.resize(ncols, nrows);

	PVCore::PVMatrix<float, int, int> mnoop,tnoop;
	mnoop.resize(nrows, ncols, 0);
	tnoop.resize(ncols, nrows);

	tbb::tick_count start,end;
	start = tbb::tick_count::now();
	matrix.transpose_to(transp);
	end = tbb::tick_count::now();

	double opt_time = (end-start).seconds();

	start = tbb::tick_count::now();
	mnoop.transpose_to(tnoop);
	end = tbb::tick_count::now();

	double time = (end-start).seconds();

	std::stringstream ss;
	ss << "transposition_speedup_" << nrows << "_" << ncols;

	PV_STAT_SPEEDUP(ss.str(), time / opt_time);

	return 0;
}
