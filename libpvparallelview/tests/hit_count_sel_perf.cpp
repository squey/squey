
#include <QCoreApplication>

#include "common.h"

#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/picviz_bench.h>

#include <picviz/PVPlotted.h>

#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVSelectionGenerator.h>
#include <pvparallelview/PVHitGraphBlocksManager.h>
#include <pvparallelview/PVZoneTree.h>

#define NBLOCKS 2

int main(int argc, char **argv)
{
	set_extra_param(1, "axis");

	printf("init_cuda: before\n");
	BENCH_START(init);
	PVParallelView::common::init_cuda();
	BENCH_STOP(init);
	printf("init_cuda in %g sec\n", BENCH_END_TIME(init));

	QCoreApplication app(argc, argv);

	PVParallelView::PVLibView* lv = create_lib_view_from_args(argc, argv);

	if (lv == nullptr) {
		return 1;
	}

	int pos = extra_param_start_at();

	const PVCol axis = atoi(argv[pos]);

	Picviz::PVSelection sel_ref, sel_sse, sel_inv, sel_sse_inv;
	sel_ref.select_all();
	sel_inv.select_all();
	sel_sse.select_all();
	sel_sse_inv.select_all();
	sel_ref.select_none();
	sel_inv.select_none();
	sel_sse.select_none();
	sel_sse_inv.select_none();

	PVParallelView::PVZoneTree& zt = *new PVParallelView::PVZoneTree();
	PVParallelView::PVZonesManager& zm = lv->get_zones_manager();
	const uint32_t *plotted = Picviz::PVPlotted::get_plotted_col_addr(zm.get_uint_plotted(), zm.get_number_rows(), axis);
	PVParallelView::PVHitGraphBlocksManager manager(zt, plotted, zm.get_number_rows(), NBLOCKS, sel_ref);
	manager.change_and_process_view(0, 2, 0.5f);

	uint32_t max_count = 0;
	const uint32_t* buffer_bg = manager.buffer_bg();
	for (size_t i = 0; i < 2048 * NBLOCKS; i++) {
		const uint32_t v = buffer_bg[i];
		if (v > max_count) {
			max_count = v;
		}
	}

	std::cout << max_count << std::endl;

	QRectF rect(0, 0, max_count, 0x6FFFFFFFU);
	PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect_serial(manager, rect, max_count, sel_ref);
	PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect_serial_invariant(manager, rect, max_count, sel_inv);
	PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect_sse(manager, rect, max_count, sel_sse);
	PVParallelView::PVSelectionGenerator::compute_selection_from_hit_count_view_rect_sse_invariant_omp(manager, rect, max_count, sel_sse_inv);

	std::cout << "Number of selected lines (ref): " << sel_ref.get_number_of_selected_lines_in_range(0, zm.get_number_rows()) << std::endl;
	std::cout << "Number of selected lines (invariant): " << sel_inv.get_number_of_selected_lines_in_range(0, zm.get_number_rows()) << std::endl;
	std::cout << "Number of selected lines (sse): " << sel_sse.get_number_of_selected_lines_in_range(0, zm.get_number_rows()) << std::endl;
	std::cout << "Number of selected lines (sse-invariant): " << sel_sse_inv.get_number_of_selected_lines_in_range(0, zm.get_number_rows()) << std::endl;

	bool valid = (sel_ref == sel_inv);
	PV_ASSERT_VALID(valid);
	valid = (sel_ref == sel_sse);
	PV_ASSERT_VALID(valid);
	valid = (sel_ref == sel_sse_inv);
	PV_ASSERT_VALID(valid);

	return 0;
}
