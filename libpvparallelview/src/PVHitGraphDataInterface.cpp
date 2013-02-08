#include <pvparallelview/PVHitGraphDataInterface.h>


//
// PVHitGraphData
//

PVParallelView::PVHitGraphDataInterface::PVHitGraphDataInterface()
{
	buffer_all().set_zero();
	buffer_sel().set_zero();
}

PVParallelView::PVHitGraphDataInterface::~PVHitGraphDataInterface()
{
}

void PVParallelView::PVHitGraphDataInterface::shift_left(int n)
{
	buffer_all().shift_left(n);
	buffer_sel().shift_left(n);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(int n)
{
	buffer_all().shift_right(n);
	buffer_sel().shift_right(n);
}

void PVParallelView::PVHitGraphDataInterface::process_allandsel(PVZoneTree const& zt, uint32_t const* col_plotted, PVRow const nrows, uint32_t const y_min, int const zoom, int const block_start, int const nblocks, Picviz::PVSelection const& sel)
{
	process_all(zt, col_plotted, nrows, y_min, zoom, block_start, nblocks);
	process_sel(zt, col_plotted, nrows, y_min, zoom, block_start, nblocks, sel);
}
