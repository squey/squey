/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_bench.h>
#include <inendi/PVSelection.h>

#include <omp.h>

using Inendi::PVSelection;

static constexpr size_t SELECTION_COUNT = 100000;

using sels_t = std::vector<PVSelection>;

void reset_sels(sels_t& sels)
{
	for (size_t i = 0; i < sels.size(); i++) {
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

	{
		sels_t sels(n, PVSelection(SELECTION_COUNT));
		reset_sels(sels);

		PVSelection& first_sel = sels[0];

		BENCH_START(serial);
		for (size_t i = 1; i < n; i++) {
			first_sel |= sels[i];
		}
		BENCH_END(serial, "serial", n, sizeof(uint32_t)*first_sel.chunk_count(), 1, sizeof(uint32_t)*first_sel.chunk_count());
	}

	{
		sels_t sels(n, PVSelection(SELECTION_COUNT));
		reset_sels(sels);

		PVSelection& first_sel_omp = sels[0];

		BENCH_START(omp);
		#pragma omp parallel for num_threads(6)
		for (size_t c = 0; c < first_sel_omp.chunk_count(); c++) {
			for (size_t i = 0; i < n; i++) {
				first_sel_omp.get_buffer()[c] |= sels[i].get_buffer()[c];
			}
		}
		BENCH_END(omp, "omp", n, sizeof(uint32_t)*first_sel_omp.chunk_count(), 1, sizeof(uint32_t)*first_sel_omp.chunk_count());
	}

	return 0;
}
