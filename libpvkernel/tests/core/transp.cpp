/**
 * \file transp.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>

#include <tbb/tick_count.h>

#include <iostream>

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " height width" << std::endl;
		return 0;
	}

	size_t nrows = atol(argv[1]);
	size_t ncols = atol(argv[2]);

	PVCore::PVMatrix<float, uint32_t, uint32_t> matrix,transp;
	matrix.resize(10, 10);
	transp.resize(10, 10);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			matrix.set_value(i, j, i);
			std::cout << matrix.at(i, j) << "\t";
		}
		std::cout << std::endl;
	}
	matrix.transpose_to(transp);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << transp.at(i, j) << "\t";
		}
		std::cout << std::endl;
	}

	matrix.resize(nrows, ncols, 0);
	transp.resize(ncols, nrows);

	PVCore::PVMatrix<float, int, int> mnoop,tnoop;
	mnoop.resize(nrows, ncols, 0);
	tnoop.resize(ncols, nrows);

	tbb::tick_count start,end;
	start = tbb::tick_count::now();
	matrix.transpose_to(transp);
	end = tbb::tick_count::now();

	std::cout << "Transposition time (optimised): " << (end-start).seconds() << std::endl;

	start = tbb::tick_count::now();
	mnoop.transpose_to(tnoop);
	end = tbb::tick_count::now();

	std::cout << "Transposition time: " << (end-start).seconds() << std::endl;

	return 0;
}
