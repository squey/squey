
#include <QApplication>

#include "common.h"

#include <pvkernel/core/picviz_bench.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVHitCountView.h>

int main(int argc, char **argv)
{
	set_extra_param(1, "axis");

	printf("init_cuda: before\n");
	BENCH_START(init);
	PVParallelView::common::init_cuda();
	BENCH_STOP(init);
	printf("init_cuda in %g sec\n", BENCH_END_TIME(init));

	QApplication app(argc, argv);

	PVParallelView::PVLibView* lv = create_lib_view_from_args(argc, argv);
	lv->lib_view()->get_real_output_selection().select_all();

	if (lv == nullptr) {
		return 1;
	}

	int pos = extra_param_start_at();

	PVCol axis = atoi(argv[pos]);

	PVParallelView::PVHitCountView *hcv = lv->create_hit_count_view(axis);

	hcv->resize(600, 400);
	hcv->show();
	hcv->setWindowTitle("Hit Count View Test");

	app.exec();

	return 0;
}

