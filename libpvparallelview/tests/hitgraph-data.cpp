
#include <pvkernel/core/picviz_bench.h>
#include <pvkernel/core/PVHardwareConcurrency.h>

#include <picviz/PVPlotted.h>

#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVHitGraphDataOMP.h>
#include <pvparallelview/PVHitGraphDataZTRight.h>

#include "common.h"

#include <iostream>

bool compare(uint32_t const* ref, uint32_t const* tab, int block_count)
{
	bool ret = true;
	for(int i = 0; i < BUFFER_SIZE * block_count; ++i) {
		if (tab[i] != ref[i]) {
			std::cerr << "differs at " << i
			          << ": " << tab[i] << " instead of " << ref[i]
			          << std::endl;
			ret = false;
		}
	}
	return ret;
}

typedef enum {
	ARG_COL = 0,
	ARG_MIN,
	ARG_ZOOM,
} EXTRA_ARG;

int main(int argc, char **argv)
{
	set_extra_param(3, "col y_min zoom");

	Picviz::PVPlotted::uint_plotted_table_t plotted;
	PVCol col_count;
	PVRow row_count;

	std::cout << "loading data" << std::endl;
	if (false == create_plotted_table_from_args(plotted, row_count, col_count, argc, argv)) {
		exit(1);
	}

	int pos = extra_param_start_at();

	int col = atoi(argv[pos + ARG_COL]);
	uint64_t y_min = atol(argv[pos + ARG_MIN]);
	int zoom = atol(argv[pos + ARG_ZOOM]);
	
	uint64_t y_max = y_min + (1UL << (32 - zoom));

	PVParallelView::PVZoneProcessing zp(plotted, row_count, col, col + 1);
	PVParallelView::PVZoneTree& zt = *new PVParallelView::PVZoneTree(zp);

	const uint32_t *col_y1 = zp.get_plotted_col_a();
	const uint32_t *col_y2 = zp.get_plotted_col_b();

	int buffer_size = 1024;

	Picviz::PVSelection selection;

	std::cout << "start test" << std::endl;

	PVParallelView::PVHitGraphDataOMP lib_omp;
	lib_omp.process(zt, col_y2, row_count, 0, 

	return 0;
}
