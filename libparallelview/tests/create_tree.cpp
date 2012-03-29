#include <pvkernel/core/picviz_bench.h>
#include <picviz/PVPlotted.h>
#include <pvparallelview/PVZoneTree.h>
#include <pvparallelview/PVTools.h>

#include <iostream>

#include <QApplication>
#include <QVector>

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " plotted_file" << std::endl;
		return 1;
	}

	QApplication app(argc, argv);

	PVCol ncols;
	Picviz::PVPlotted::plotted_table_t plotted;
	PVLOG_INFO("Loading plotted...\n");
	if (!Picviz::PVPlotted::load_buffer_from_file(plotted, ncols, true, QString(argv[1]))) {
		std::cerr << "Unable to load plotted !" << std::endl;
		return 1;
	}
	PVRow nrows = plotted.size()/ncols;
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
		PVParallelView::PVZoneTree<std::vector<PVRow> >* ztree = new PVParallelView::PVZoneTree<std::vector<PVRow> >(0, 1);
		ztree->set_trans_plotted(norm_plotted, nrows, ncols);

		PVLOG_INFO("Zone tree creation...\n");
		BENCH_START(org);
		ztree->process_sse();
		BENCH_END_TRANSFORM(org, "sse", nrows*2, sizeof(float));

		Picviz::PVSelection sel;
		sel.select_all();
		//sel.set_line(40, true);
		PVLOG_INFO("Sub-tree from selection creation...\n");
		PVParallelView::PVZoneTree<std::vector<PVRow> >* ztree_sel = ztree->filter_by_sel<false>(sel);
		delete ztree_sel;
		ztree_sel = ztree->filter_by_sel<true>(sel);

		// Display this tree
		ztree->display("zone-sse", plotted);
		ztree_sel->display("zone-sse-sel-first", plotted);
	}

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


	return app.exec();
}
