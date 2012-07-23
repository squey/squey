/**
 * \file selection_red.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVSelection.h>

#include <omp.h>

using Picviz::PVSelection;

void reset_sels(PVSelection* sels, const size_t n)
{
	for (size_t i = 0; i < n; i++) {
		sels[i].select_none();
		sels[i].set_line(i, true);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " n" << std::endl;
	}

	size_t n = atoll(argv[1]);
	PVSelection* sels = new PVSelection[n];
	reset_sels(sels, n);

	PVSelection& first_sel = sels[0];
	BENCH_START(serial);
	for (size_t i = 1; i < n; i++) {
		first_sel |= sels[i];
	}
	BENCH_END(serial, "serial", n, sizeof(uint32_t)*PICVIZ_SELECTION_NUMBER_OF_CHUNKS, 1, sizeof(uint32_t)*PICVIZ_SELECTION_NUMBER_OF_CHUNKS);

	PVSelection sel_ref = first_sel;

	delete [] sels;
	sels = new PVSelection[n];
	reset_sels(sels, n);
	PVSelection& first_sel_omp = sels[0];

	BENCH_START(omp);
#pragma omp parallel for num_threads(6)
	for (size_t c = 0; c < PICVIZ_SELECTION_NUMBER_OF_CHUNKS; c++) {
		for (size_t i = 0; i < n; i++) {
			first_sel_omp.get_buffer()[c] |= sels[i].get_buffer()[c];
		}
	}
	BENCH_END(omp, "omp", n, sizeof(uint32_t)*PICVIZ_SELECTION_NUMBER_OF_CHUNKS, 1, sizeof(uint32_t)*PICVIZ_SELECTION_NUMBER_OF_CHUNKS);


	return 0;
}
