/**
 * \file both_views.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <pvparallelview/common.h>
#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCode.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVLinesView.h>

#include <pvparallelview/PVLibView.h>

#include <pvbase/general.h>

#include <QCoreApplication>

#include "common.h"

void count_y2_seq(uint32_t* count_buffer, PVParallelView::PVZoneTree const& zt, const uint32_t* plotted, const uint32_t zoom, const uint32_t y_min)
{
	assert(zoom >=1 && zoom <= 10);
	const int idx_shift = (32 - NBITS_INDEX) - zoom;
	const uint32_t idx_mask = (1 << NBITS_INDEX) - 1;
	const uint32_t mask_branch = ((1<<zoom) - 1) << (32 - zoom);

	const uint32_t branch_start = y_min & mask_branch;

	PVParallelView::PVBCode b_start;
	b_start.s.r = branch_start;
	b_start.s.l = 0;

	PVParallelView::PVBCode b_end;
	b_end.s.r = branch_start | (~mask_branch);
	b_end.s.l = (1<<NBITS_INDEX)-1;

	// The advantage of doing this on the right axis is that our
	// indexes are sequentials !
	BENCH_START(bcount);
	for (uint32_t b = b_start.int_v; b < b_end.int_v; b++) {
		PVParallelView::PVZoneTree::PVBranch const& branch = zt.get_branch(b);
		PVRow const* rows = branch.p;
		for (size_t i = 0; i < branch.count; i++) {
			const PVRow r = rows[i];
			const uint32_t y_plotted = plotted[r];

			const uint32_t idx = (y_plotted >> idx_shift) & idx_mask;
			count_buffer[idx]++;
		}
	}
	BENCH_END(bcount, "count-seq", b_end.int_v-b_start.int_v, 1023, sizeof(uint32_t), 1);
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QCoreApplication app(argc, argv);

	PVCol ncols;
	PVRow nrows;

	Picviz::PVPlotted::uint_plotted_table_t plotted;
	if (!create_plotted_table_from_args(plotted, nrows, ncols, argc, argv)) {
		return 1;
	}

	PVCol col = 1;
	assert(col >= 1);

	// Number of rows is aligned on a multiple of 4
	uint32_t const* plotted_col = &plotted[col*((nrows/4)*4)];

	PVParallelView::PVZoneProcessing zp(plotted, nrows, 0, 1);
	PVParallelView::PVZoneTree &zt = *new PVParallelView::PVZoneTree();
	zt.process(zp);

	// Computations
	// Level 0
#define SIZE_RED ((1<<NBITS_INDEX)-1)
	uint32_t* count_buffer;
	posix_memalign((void**) &count_buffer, 16, sizeof(uint32_t)*SIZE_RED);
	BENCH_START(zoom0);
	for (uint32_t i = 0; i < SIZE_RED; i++) {
		count_buffer[i] = zt.get_right_axis_count(i);
	}
	BENCH_END(zoom0, "zoom0", 1, 1, 1, 1);

	memset(count_buffer, 0, sizeof(uint32_t)*SIZE_RED);
	count_y2_seq(count_buffer, zt, plotted_col, 9, 0);

	PVParallelView::common::release();


	return 0;
}
