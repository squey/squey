/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QApplication>

#include "common.h"

#include <pvkernel/core/inendi_bench.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVScatterView.h>

int main(int argc, char **argv)
{
	set_extra_param(1, "axis");

	printf("init_cuda: before\n");
	BENCH_START(init);
	PVParallelView::common::RAII_cuda_init cuda_resources;
	BENCH_STOP(init);
	printf("init_cuda in %g sec\n", BENCH_END_TIME(init));

	QApplication app(argc, argv);

	PVParallelView::PVLibView* lv = create_lib_view_from_args(argc, argv);

	if (lv == nullptr) {
		return 1;
	}

	lv->lib_view()->apply_filter_named_select_all();
	lv->lib_view()->process_from_selection();

	int pos = extra_param_start_at();

	PVCol axis = atoi(argv[pos]);

	PVParallelView::PVScatterView* sv = lv->create_scatter_view(axis);

	sv->setWindowTitle("Scatter View Test");
	sv->resize(1024, 768);
	sv->show();

	app.exec();

	return 0;
}





