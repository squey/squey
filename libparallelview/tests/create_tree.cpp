#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVTools.h>
#include <pvparallelview/PVHSVColor.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QVector>

#include <tbb/scalable_allocator.h>
#include <tbb/concurrent_vector.h>
#include <boost/unordered_map.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVRow nrows, PVCol ncols)
{
	srand(time(NULL));
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
		p.push_back(0.5);
	}
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
}

void test(Picviz::PVPlotted::plotted_table_t& plotted, PVRow nrows, PVCol ncols, PVParallelView::PVHSVColor* colors, PVParallelView::PVBCICode* bci_codes)
{
	plotted_int_t norm_plotted;
	//PVLOG_INFO("Normalizing to 32-bit unsigned integers...\n");
	PVParallelView::PVTools::norm_int_plotted(plotted, norm_plotted, ncols);
	//PVLOG_INFO("Done !\n");

	{
		MEM_START(serial);
		PVParallelView::PVZoneTree<std::vector<PVRow> >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow> >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		BENCH_START(org);
		ztree->process();
		BENCH_END_TRANSFORM(org, "org", nrows*2, sizeof(float));
		MEM_END(serial, "org");

		{
		MEM_START(serial);
		BENCH_START(org);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		BENCH_END(org, "org colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "org colors");
		}

		//ztree->display("zone", plotted);
		delete ztree;
	}
	std::cout << "---" << std::endl;

	{
		MEM_START(serial);
		PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		BENCH_START(sse);
		ztree->process_sse();

		BENCH_END_TRANSFORM(sse, "serial sse + std::vector", nrows*2, sizeof(float));
		MEM_END(serial, "serial sse + std::vector");

		{
		MEM_START(serial);
		BENCH_START(sse);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		picviz_verify(sizeof(PVParallelView::PVHSVColor) == 1);
		BENCH_END(sse, "serial sse + std::vector colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "serial sse + std::vector colors");
		}

		//ztree->display("zone", plotted);
		delete ztree;
	}
	std::cout << "---" << std::endl;

	{
		MEM_START(serial);
		PVParallelView::PVZoneTreeNoAlloc* ztree = new PVParallelView::PVZoneTreeNoAlloc(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		BENCH_START(sse);
		ztree->process_sse();

		BENCH_END_TRANSFORM(sse, "serial sse + noalloc", nrows*2, sizeof(float));
		MEM_END(serial, "serial sse + noalloc");

		{
		MEM_START(serial);
		BENCH_START(sse);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		picviz_verify(sizeof(PVParallelView::PVHSVColor) == 1);
		BENCH_END(sse, "serial sse + noalloc colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "serial sse + noalloc colors");
		}

		//ztree->display("zone-noalloc", plotted);
		delete ztree;
	}
	std::cout << "---" << std::endl;

	
	{
		MEM_START(serial);
		PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		//PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(sse);
		ztree->process_omp_sse();
		BENCH_END_TRANSFORM(sse, "omp sse + std::vector", 1, 1);
		MEM_END(serial, "omp sse + std::vector");

		{
		MEM_START(serial);
		BENCH_START(sse);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		picviz_verify(sizeof(PVParallelView::PVHSVColor) == 1);
		BENCH_END(sse, "omp sse + std::vector colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "omp sse + std::vector colors");
		}

		//ztree->display("zone-omp", plotted);
		delete ztree;
	}
	std::cout << "---" << std::endl;


	{
		MEM_START(serial);
		PVParallelView::PVZoneTreeNoAlloc* ztree = new PVParallelView::PVZoneTreeNoAlloc(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		BENCH_START(sse);
		ztree->process_omp_sse();
		BENCH_END_TRANSFORM(sse, "omp sse + noalloc", 1, 1);
		MEM_END(serial, "omp sse + noalloc");
		//ztree->display("zone-omp", plotted);

		{
		MEM_START(serial);
		BENCH_START(sse);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		picviz_verify(sizeof(PVParallelView::PVHSVColor) == 1);
		BENCH_END(sse, "omp sse + noalloc colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "omp sse + noalloc colors");
		}

		delete ztree;
	}
	std::cout << "---" << std::endl;

	/*{
		MEM_START(serial);
		typedef boost::unordered_multimap<unsigned int, PVRow, boost::hash<unsigned int>, std::equal_to<unsigned int>, std::allocator<std::pair<const unsigned int, PVRow> > > boost_unordered_map;
		typedef PVParallelView::PVZoneTreeUnorderedMap<boost_unordered_map> boost_tree;
		boost_tree* ztree = new boost_tree(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);
		BENCH_START(b);
		ztree->process_boost();
		BENCH_END(b, "boost", 1, 1, 1, 1);

		//ztree->process();
		MEM_END(serial, "boost");
		delete ztree;
	}
	std::cout << "---" << std::endl;*/

	{
		MEM_START(serial);
		typedef tbb::concurrent_vector< /*std::pair<unsigned int ,*/ PVRow/*>*/ > cv_t;
		typedef PVParallelView::PVZoneTree<cv_t> tbb_concurrent_vector_tree;
		tbb_concurrent_vector_tree* ztree = new tbb_concurrent_vector_tree(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);
		BENCH_START(b);
		ztree->process_tbb_concurrent_vector();

		BENCH_END(b, "tbb concurrent vector", 1, 1, 1, 1);

		//ztree->process();
		MEM_END(serial, "tbb concurrent vector");

		{
		MEM_START(serial);
		BENCH_START(b);
		size_t nb_codes = ztree->browse_tree_bci(colors, bci_codes);
		picviz_verify(sizeof(PVParallelView::PVHSVColor) == 1);
		BENCH_END(b, "tbb concurrent vector colors", nb_codes, sizeof(PVRow), nb_codes, sizeof(PVParallelView::PVBCICode));
		MEM_END(serial, "tbb concurrent vector colors");
		}

		delete ztree;
	}
	std::cout << "---" << std::endl;


	/*{
		Picviz::PVSelection sel;
		//sel.select_all();
		sel.set_line(40, true);
		PVLOG_INFO("Sub-tree from selection creation...\n");
		PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >* ztree_sel = ztree->filter_by_sel<false>(sel);
		delete ztree_sel;
		ztree_sel = ztree->filter_by_sel<true>(sel);

		// Display this tree
		ztree->display("zone-sse", plotted);
		//ztree_sel->display("zone-sse-sel-first", plotted);
	}*/

	/*{
		PVParallelView::PVZoneTree<QList<PVRow> >* ztree = new PVParallelView::PVZoneTree<QList<PVRow> >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(org);
		ztree->process();
		BENCH_END_TRANSFORM(org, "org_qlist", nrows*2, sizeof(float));

		// Display this tree
		ztree->display("zone", plotted);
	}

	{
		PVParallelView::PVZoneTree<QVector<PVRow> >* ztree = new PVParallelView::PVZoneTree<QVector<PVRow> >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(org);
		ztree->process();
		BENCH_END_TRANSFORM(org, "org_qvector", nrows*2, sizeof(float));

		// Display this tree
		ztree->display("zone", plotted);
	}*/
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
	PVLOG_INFO("Loading plotted...\n");
	QString fplotted(argv[1]);
	if (fplotted == "0") {
		PVLOG_INFO("Initialising random plotted...\n");
		if (argc < 4) {
			usage(argv[0]);
			return 1;
		}
		srand(time(NULL));
		nrows = atol(argv[2]);
		ncols = atol(argv[3]);

		PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(nrows);
		PVParallelView::PVBCICode* bci_codes = PVParallelView::PVBCICode::allocate_codes(NBUCKETS);

		std::cout << "== RANDOM DISTRIBUTED PLOTTED ==" << std::endl;
		init_rand_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes);

		std::cout << "== NORMAL DISTRIBUTED PLOTTED ==" << std::endl;
		init_normal_plotted(plotted, nrows, ncols);
		test(plotted, nrows, ncols, colors, bci_codes);

		delete [] colors;
		PVParallelView::PVBCICode::free_codes(bci_codes);
	}
	else
	{
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
	}
	//PVLOG_INFO("Plotted loaded with %u rows and %u columns.\n", nrows, ncols);

	plotted_int_t norm_plotted;
	//PVLOG_INFO("Normalizing to 32-bit unsigned integers...\n");
	PVParallelView::PVTools::norm_int_plotted(plotted, norm_plotted, ncols);
	//PVLOG_INFO("Done !\n");

	//app.exec();

	return 0;
	//return app.exec();
}
