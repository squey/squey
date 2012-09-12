/**
 * \file zoomed_zone_tree_render.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVBCIDrawingBackendCUDA.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVTools.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZonesManager.h>

#include <pvbase/general.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QString>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#define RENDERING_BITS PARALLELVIEW_ZZT_BBITS

typedef PVParallelView::PVZonesDrawing<RENDERING_BITS> zones_drawing_t;
typedef PVParallelView::PVBCICode<RENDERING_BITS> bcicode_t;

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void fdprintf(int fd, const char *format, ...)
{
	char buffer [2048];

	va_list ap;
	va_start(ap, format);
	(void) vsnprintf (buffer, 2048, format, ap);
	va_end(ap);

	(void) write (fd, buffer, strlen(buffer));
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
#if 1
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
#else
	for (PVCol j = 0; j < ncols; ++j) {
		for (PVRow i = 0; i < nrows; i++) {
			p.push_back((32. * i) / 1024.);
			// p.push_back((32. * i + .25) / 1024.);
			// p.push_back((32. * i + 0.5) / 1024.);
		}
	}
#endif
}

bool bci_cmp(const bcicode_t &a, const bcicode_t &b)
{
	return a.s.idx < b.s.idx;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols, nrows;
	Picviz::PVPlotted::plotted_table_t plotted;
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}

		ncols = atol(argv[3]);

		if (ncols < 2) {
			std::cout << "ncols must be greater or equal to 2, using 2" << std::endl;
			ncols = 2;
		}

		init_rand_plotted(plotted, nrows, ncols);
	}
	else
	{
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;

		if (nrows > PICVIZ_LINES_MAX) {
			std::cerr << "nrows is too big (max is " << PICVIZ_LINES_MAX << ")" << std::endl;
			return 1;
		}
	}

	PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);

	for(int i = 0; i < nrows; ++i) {
		colors[i] = (i*20) % 192;
	}

	bcicode_t *bcicodes_seq = new bcicode_t [NBUCKETS];
	bcicode_t *bcicodes_tbb = new bcicode_t [NBUCKETS];

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	PVParallelView::PVZonesManager &zm = *(new PVParallelView::PVZonesManager());
	zm.set_uint_plotted(norm_plotted, nrows, ncols);
	zm.update_all();

	PVParallelView::PVZoomedZoneTree::context_t zzt_ctx;
	uint64_t y_min = 0;
	uint64_t y_max = 1ULL << 32;
	uint32_t zoom = 0;

	PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree = zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(0);

	{
		// a run to allocate what has to be
		BENCH_START(allocate_run);
		zoomed_zone_tree.browse_bci_by_y1_tbb(zzt_ctx,
		                                      y_min, y_max, y_max,
		                                      zoom, 512,
		                                      colors, bcicodes_tbb);
		BENCH_END(allocate_run, "allocate_run", 1, 1, 1, 1);

		// the parallel perf run
		BENCH_START(browse_tbb);
		size_t num_tbb = zoomed_zone_tree.browse_bci_by_y1_tbb(zzt_ctx,
		                                                       y_min, y_max, y_max,
		                                                       zoom, 512,
		                                                       colors, bcicodes_tbb);
		BENCH_END(browse_tbb, "TBB browse", 1, 1, 1, 1);

		// the sequential perf run (to compare to)
		BENCH_START(browse_seq);
		size_t num_seq = zoomed_zone_tree.browse_bci_by_y1_seq(zzt_ctx,
		                                                       y_min, y_max, y_max,
		                                                       zoom, 512,
		                                                       colors, bcicodes_seq);
		BENCH_END(browse_seq, "SEQ browse", 1, 1, 1, 1);

		std::sort(bcicodes_tbb, bcicodes_tbb + num_tbb, bci_cmp);
		std::sort(bcicodes_seq, bcicodes_seq + num_seq, bci_cmp);

		size_t num = 0;

		if (num_seq != num_tbb) {
			std::cout << "Y1: seq & tbb do not return the same count; using smallest value"
			          << std::endl;
			num = std::min(num_tbb, num_seq);
		} else {
			num = num_seq;
		}

		std::cout << "Y1: memcmp: " << memcmp(bcicodes_seq, bcicodes_tbb,
		                                      sizeof(bcicode_t) * num) << std::endl;
	}

	{
		// a run to allocate what has to be
		BENCH_START(allocate_run);
		zoomed_zone_tree.browse_bci_by_y2_tbb(zzt_ctx,
		                                      y_min, y_max, y_max,
		                                      zoom, 512,
		                                      colors, bcicodes_tbb);
		BENCH_END(allocate_run, "allocate_run", 1, 1, 1, 1);

		// the parallel perf run
		BENCH_START(browse_tbb);
		size_t num_tbb = zoomed_zone_tree.browse_bci_by_y2_tbb(zzt_ctx,
		                                                       y_min, y_max, y_max,
		                                                       zoom, 512,
		                                                       colors, bcicodes_tbb);
		BENCH_END(browse_tbb, "TBB browse", 1, 1, 1, 1);

		// the sequential perf run (to compare to)
		BENCH_START(browse_seq);
		size_t num_seq = zoomed_zone_tree.browse_bci_by_y2_seq(zzt_ctx,
		                                                       y_min, y_max, y_max,
		                                                       zoom, 512,
		                                                       colors, bcicodes_seq);
		BENCH_END(browse_seq, "SEQ browse", 1, 1, 1, 1);

		std::sort(bcicodes_tbb, bcicodes_tbb + num_tbb, bci_cmp);
		std::sort(bcicodes_seq, bcicodes_seq + num_seq, bci_cmp);

		size_t num = 0;

		if (num_seq != num_tbb) {
			std::cout << "Y2: seq & tbb do not return the same count; using smallest value"
			          << std::endl;
			num = std::min(num_tbb, num_seq);
		} else {
			num = num_seq;
		}

		std::cout << "Y2: memcmp: " << memcmp(bcicodes_seq, bcicodes_tbb,
		                                      sizeof(bcicode_t) * num) << std::endl;
	}

	return 0;
}
