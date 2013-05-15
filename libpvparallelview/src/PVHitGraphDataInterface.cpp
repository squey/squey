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

void PVParallelView::PVHitGraphDataInterface::shift_left(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_left(n, alpha);
	buffer_sel().shift_zoomed_left(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_right(n, alpha);
	buffer_sel().shift_zoomed_right(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::process_all(ProcessParams const& params,
                                                          Picviz::PVSelection const& layer_sel,
                                                          Picviz::PVSelection const& sel)
{
	process_bg(params, layer_sel);
	process_sel(params, sel);
}
