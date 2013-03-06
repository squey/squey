#include <pvparallelview/PVHitGraphDataInterface.h>


//
// PVHitGraphData
//

PVParallelView::PVHitGraphDataInterface::PVHitGraphDataInterface(uint32_t nbits, uint32_t nblocks):
	_buf_all(nbits, nblocks),
	_buf_sel(nbits, nblocks)
{
	buffer_all().set_zero();
	buffer_sel().set_zero();
}

PVParallelView::PVHitGraphDataInterface::~PVHitGraphDataInterface()
{
}

void PVParallelView::PVHitGraphDataInterface::shift_left(const uint32_t n)
{
	buffer_all().shift_left(n);
	buffer_sel().shift_left(n);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(const uint32_t n)
{
	buffer_all().shift_right(n);
	buffer_sel().shift_right(n);
}

void PVParallelView::PVHitGraphDataInterface::process_allandsel(ProcessParams const& params, Picviz::PVSelection const& sel)
{
	process_all(params);
	process_sel(params, sel);
}

void PVParallelView::PVHitGraphDataInterface::process_zoom_reduction(const float alpha)
{
	buffer_all().process_zoom_reduction(alpha);
	buffer_sel().process_zoom_reduction(alpha);
}
