#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVTools.h>

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <QApplication>
#include <QVector>

#include <tbb/scalable_allocator.h>

void init_rand_plotted(Picviz::PVPlotted::plotted_table_t& p, PVCol ncols, PVRow nrows)
{
	srand(time(NULL));
	p.clear();
	p.reserve(nrows*ncols);
	for (PVRow i = 0; i < nrows*ncols; i++) {
		p.push_back((float)((double)(rand())/(double)RAND_MAX));
	}
}

void usage(const char* path)
{
	std::cerr << "Usage: " << path << " [plotted_file] [nrows] [ncols]" << std::endl;
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
		init_rand_plotted(plotted, nrows, ncols);
	}
	else
	{
		if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
			std::cerr << "Unable to load plotted !" << std::endl;
			return 1;
		}
		nrows = plotted.size()/ncols;
	}
	PVLOG_INFO("Plotted loaded with %u rows and %u columns.\n", nrows, ncols);

	plotted_int_t norm_plotted;
	PVLOG_INFO("Normalizing to 32-bit unsigned integers...\n");
	PVParallelView::PVTools::norm_int_plotted(plotted, norm_plotted, ncols);
	PVLOG_INFO("Done !\n");

	/*{
		PVParallelView::PVZoneTree<std::vector<PVRow> >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow> >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(org);
		ztree->process();
		BENCH_END_TRANSFORM(org, "org", nrows*2, sizeof(float));

		// Display this tree
		ztree->display("zone", plotted);
	}*/

	{
		MEM_START(serial);
		PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(sse);
		//ztree->process_sse();
		BENCH_END_TRANSFORM(sse, "sse", nrows*2, sizeof(float));
		MEM_END(serial, "serial sse + std::vector");
		//ztree->display("zone", plotted);
		delete ztree;
	}


	{
		MEM_START(serial);
		PVParallelView::PVZoneTreeNoAlloc* ztree = new PVParallelView::PVZoneTreeNoAlloc(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(sse);
		//ztree->process_sse();
		BENCH_END_TRANSFORM(sse, "noalloc-sse", nrows*2, sizeof(float));
		MEM_END(serial, "serial sse + noalloc");
		//ztree->display("zone-noalloc", plotted);
		delete ztree;
	}

	
	{
		MEM_START(serial);
		PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow, tbb::scalable_allocator<PVRow> > >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		//ztree->process_omp_sse();
		MEM_END(serial, "omp sse + std::vector");
		//ztree->display("zone-omp", plotted);
	}

	{
		MEM_START(serial);
		PVParallelView::PVZoneTreeNoAlloc* ztree = new PVParallelView::PVZoneTreeNoAlloc(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree noalloc creation...\n");
		ztree->process_omp_sse();
		MEM_END(serial, "omp sse + noalloc");
		//ztree->display("zone-omp", plotted);
		delete ztree;
	}


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


	return 0;
	//return app.exec();
}
