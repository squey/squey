/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/inendi_bench.h>
#include <inendi/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVZoneTreeNoAlloc.h>
#include <pvparallelview/PVTools.h>
#include <pvkernel/core/PVHSVColor.h>
#include <pvparallelview/simple_lines_int_view.h>

#include <pvbase/general.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QVector>
#include <QMainWindow>
#include <QString>

#include <tbb/scalable_allocator.h>
#include <tbb/concurrent_vector.h>
#include <boost/unordered_map.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

void show_codes(QString const& title, PVParallelView::PVBCICode<NBITS_INDEX>* codes, size_t n)
{
	QMainWindow* mw = new QMainWindow();
	mw->setWindowTitle(title);
	SLIntView* v = new SLIntView(mw);
	v->set_size(1024, 1024);
	v->set_ortho(1, 1024);

	std::vector<int32_t>& pts = *(new std::vector<int32_t>);
	std::vector<PVRGB>& colors = *(new std::vector<PVRGB>);
	pts.reserve(n*4);
	colors.reserve(n);
	PVRGB rgb;
	rgb.int_v = 0;
	for (size_t i = 0; i < n; i++) {
		PVParallelView::PVBCICode<NBITS_INDEX> c = codes[i];
		c = codes[i];
		pts.push_back(0); pts.push_back(c.s.l);
		pts.push_back(1); pts.push_back(c.s.r);

		PVCore::PVHSVColor hsv(c.s.color);
		hsv.to_rgb((uint8_t*) &rgb);
		colors.push_back(rgb);
	}
	v->set_points(pts);
	v->set_colors(colors);
	mw->setCentralWidget(v);
	mw->resize(v->sizeHint());
	mw->show();
}

void init_rand_plotted(Inendi::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

void init_normal_plotted(Inendi::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	// Generator engine
	boost::mt19937 rand_gen(boost::mt19937(time(0)));

	// Normal distribution
	typedef boost::random::normal_distribution<double> normal_dist_t;
	double mean = 0.5;
	double variance = 0.01;
	normal_dist_t normal_dist(mean, variance);

	// Fill plotted
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows; i++) {
		p.push_back(normal_dist(rand_gen));
	}
	for (PVRow i = 0; i < nrows*(ncols-1); i++) {
		if (i &1)
			p.push_back(0.25);
		else
			p.push_back(0.75);
	}
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void test(
	Inendi::PVPlotted::plotted_table_t& plotted,
	PVRow nrows,
	PVCol ncols,
	PVCore::PVHSVColor* /*colors*/,
	PVParallelView::PVBCICode<NBITS_INDEX>* /*bci_codes*/,
	PVParallelView::PVBCICode<NBITS_INDEX>* /*bci_codes_ref*/
)
{
	Inendi::PVPlotted::uint_plotted_table_t norm_plotted;
	Inendi::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	{
		PVParallelView::PVZoneTree* ztree = new PVParallelView::PVZoneTree();
		//ztree->set_trans_plotted(norm_plotted, nrows, ncols);
		PVParallelView::PVZoneProcessing zp(norm_plotted, nrows, 0, 1);

		{
		BENCH_START(sse);
		ztree->process_tbb_sse_treeb(zp);
		BENCH_END_TRANSFORM(sse, "process_tbb_sse_treeb", 1, 1);
		}

		bool sorted = true;
		for (uint32_t b = 0; b < NBUCKETS; b++) {
			sorted = true;
			uint32_t last_value = 0;
			for (size_t i = 0; i < ztree->get_branch_count(b); i++) {
				sorted &= ztree->get_branch_element(b, i) >= last_value;
				last_value = ztree->get_branch_element(b, i);
				//PVLOG_INFO("treeb[%d]=%d sorted=%d\n", b, treeb[b].p[i], sorted);
			}
			if (ztree->get_branch_count(b)) {
				//PVLOG_INFO("---\n");
			}
		}
		PVLOG_INFO("sorted=%d\n", sorted);

		delete ztree;
	}
	std::cout << "---" << std::endl;

}

int main(int argc, char** argv)
{
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols, nrows;
	Inendi::PVPlotted::plotted_table_t plotted;
	//PVLOG_INFO("Loading plotted...\n");
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		//PVLOG_INFO("Initialising random plotted...\n");
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);

		ncols = atol(argv[3]);

		PVCore::PVHSVColor* colors = PVCore::PVHSVColor::init_colors(nrows);
		PVParallelView::PVBCICode<NBITS_INDEX>* bci_codes_ref = PVParallelView::PVBCICode<NBITS_INDEX>::allocate_codes(NBUCKETS);
		PVParallelView::PVBCICode<NBITS_INDEX>* bci_codes = PVParallelView::PVBCICode<NBITS_INDEX>::allocate_codes(NBUCKETS);

		std::cout << "== RANDOM DISTRIBUTED PLOTTED ==" << std::endl;
		init_rand_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes, bci_codes_ref);

		std::cout << "== NORMAL DISTRIBUTED PLOTTED ==" << std::endl;
		init_normal_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes, bci_codes_ref);

		delete [] colors;
		PVParallelView::PVBCICode<NBITS_INDEX>::free_codes(bci_codes_ref);
		PVParallelView::PVBCICode<NBITS_INDEX>::free_codes(bci_codes);

	}
	else
	{
		if (!Inendi::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
	}
	//PVLOG_INFO("Plotted loaded with %u rows and %u columns.\n", nrows, ncols);

	Inendi::PVPlotted::uint_plotted_table_t norm_plotted;
	//PVLOG_INFO("Normalizing to 32-bit unsigned integers...\n");
	Inendi::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
	//PVLOG_INFO("Done !\n");

	app.exec();

	return 0;
	//return app.exec();
}
