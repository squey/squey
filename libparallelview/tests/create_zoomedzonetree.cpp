#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVZoneProcessing.h>
#include <pvparallelview/PVZoomedZoneTree.h>
#include <pvparallelview/PVTools.h>
#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/simple_lines_int_view.h>

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

#include <stdarg.h>
#include <stdlib.h>

enum {
	MODE_SEQ,
	MODE_OMP,
	MODE_SEQ_ZT,
	MODE_OMP_ZT,
	MODE_FILE
};

int mode_value;

void show_codes(QString const& title, PVParallelView::PVBCICode* codes, size_t n)
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
		PVParallelView::PVBCICode c = codes[i];
		c = codes[i];
		pts.push_back(0); pts.push_back(c.s.l);
		pts.push_back(1); pts.push_back(c.s.r);

		PVParallelView::PVHSVColor hsv(c.s.color);
		hsv.to_rgb((uint8_t*) &rgb);
		colors.push_back(rgb);
	}
	v->set_points(pts);
	v->set_colors(colors);
	mw->setCentralWidget(v);
	mw->resize(v->sizeHint());
	mw->show();
}

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	//srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

void init_normal_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
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
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl << std::endl;
	std::cerr << "if plotted_file == 0, test ZZT::process_seq" << std::endl;
	std::cerr << "if plotted_file == 1, test ZZT::process_omp" << std::endl;
	std::cerr << "if plotted_file == 2, test ZZT::process_seq_zt" << std::endl;
	std::cerr << "if plotted_file == 3, test ZZT::process_omp_zt" << std::endl;
}

void memprintf(const char *text, size_t mem)
{
	if (mem < 1024) {
		printf ("%s: %lu o\n", text, mem);
	} else if (mem < (1024 * 1024)) {
		printf ("%s: %.3f Kio\n", text, mem / 1024.);
	} else if (mem < (1024 * 1024 * 1024)) {
		printf ("%s: %.3f Mio\n", text, mem / (1024. * 1024.));
	} else {
		printf ("%s: %.3f Gio\n", text, mem / (1024. * 1024. * 1024.));
	}
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

void test(
	Picviz::PVPlotted::plotted_table_t& plotted,
	PVRow nrows,
	PVCol ncols,
	PVParallelView::PVHSVColor* colors,
	PVParallelView::PVBCICode* bci_codes,
	PVParallelView::PVBCICode* bci_codes_ref
)
{
	(void) bci_codes_ref;

	std::cout << "== test ==" << std::endl;

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);

	{
		PVParallelView::PVZoomedZoneTree* zzt = new PVParallelView::PVZoomedZoneTree(8);
		PVParallelView::PVZoneProcessing zp(norm_plotted, nrows, 0, 1);
		PVParallelView::PVZoneTree *zt = 0;

		if ((mode_value == MODE_SEQ_ZT) || (mode_value == MODE_OMP_ZT)) {
			zt = new PVParallelView::PVZoneTree();
			std::cout << "== ZT::process ==" << std::endl;
			BENCH_START(process);
			zt-> process(zp);
			BENCH_END_TRANSFORM(process, "ZT::process", 1, 1);
		}

		std::cout << "== ZZT::process ==" << std::endl;
		{
			BENCH_START(process);
			if (mode_value == MODE_SEQ) {
				zzt->process_seq(zp);
			} else if (mode_value == MODE_OMP) {
				zzt->process_omp(zp);
			} else if (mode_value == MODE_SEQ_ZT) {
				zzt->process_seq_from_zt(zp, *zt);
			} else if (mode_value == MODE_OMP_ZT) {
				zzt->process_omp_from_zt(zp, *zt);
			}
			BENCH_END_TRANSFORM(process, "ZZT::process", 1, 1);
			memprintf("memory", zzt->memory());
		}

		size_t nb_codes;
		{
			BENCH_START(browse);
			nb_codes = zzt->browse_tree_bci_by_y1(0, UINT_MAX, colors, bci_codes);
			BENCH_END_TRANSFORM(browse, "ZZT::browse_tree_bci_by_y1", 1, 1);
			std::cout << "browse found " << nb_codes << " element(s)" << std::endl;
		}

		(void) nb_codes;
		// show_codes("serial", bci_codes, nb_codes);
		delete zzt;
		if (zt != 0) {
			delete zt;
		}
	}
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
	//PVLOG_INFO("Loading plotted...\n");
	QString fplotted(argv[1]);

	if (fplotted == "0") {
		mode_value = MODE_SEQ;
	} else if (fplotted == "1") {
		mode_value = MODE_OMP;
	} else if (fplotted == "2") {
		mode_value = MODE_SEQ_ZT;
	} else if (fplotted == "3") {
		mode_value = MODE_OMP_ZT;
	} else {
		mode_value = MODE_FILE;
	}

	if (mode_value != MODE_FILE) {
		//PVLOG_INFO("Initialising random plotted...\n");
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		//srand(time(NULL));
		srand(0);
		nrows = atol(argv[2]);
		ncols = atol(argv[3]);

		memprintf("real data size", nrows * sizeof(PVParallelView::PVQuadTreeEntry));
		PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(nrows);
		PVParallelView::PVBCICode* bci_codes_ref = PVParallelView::PVBCICode::allocate_codes(NBUCKETS);
		PVParallelView::PVBCICode* bci_codes = PVParallelView::PVBCICode::allocate_codes(NBUCKETS);

		std::cout << "== RANDOM DISTRIBUTED PLOTTED ==" << std::endl;
		init_rand_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes, bci_codes_ref);

		std::cout << "== NORMAL DISTRIBUTED PLOTTED ==" << std::endl;
		init_normal_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes, bci_codes_ref);

		delete [] colors;
		PVParallelView::PVBCICode::free_codes(bci_codes_ref);
		PVParallelView::PVBCICode::free_codes(bci_codes);
	} else {
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
		std::cout << "Plotted loaded" << std::endl;

		memprintf("real data size", nrows * sizeof(PVParallelView::PVQuadTreeEntry));
		PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(nrows);
		PVParallelView::PVBCICode* bci_codes_ref = PVParallelView::PVBCICode::allocate_codes(NBUCKETS);
		PVParallelView::PVBCICode* bci_codes = PVParallelView::PVBCICode::allocate_codes(NBUCKETS);

		std::cout << "== " << argv[1] << " ==" << std::endl;
		test(plotted, nrows, ncols, colors, bci_codes, bci_codes_ref);
	}
	//PVLOG_INFO("Plotted loaded with %u rows and %u columns.\n", nrows, ncols);

	Picviz::PVPlotted::uint_plotted_table_t norm_plotted;
	//PVLOG_INFO("Normalizing to 32-bit unsigned integers...\n");
	Picviz::PVPlotted::norm_int_plotted(plotted, norm_plotted, ncols);
	//PVLOG_INFO("Done !\n");

	// app.exec();

	return 0;
	//return app.exec();
}
