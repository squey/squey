#include <pvparallelview/PVHitGraphDataInterface.h>


//
// PVHitGraphData
//

PVParallelView::PVHitGraphDataInterface::PVHitGraphDataInterface(uint32_t nbits, uint32_t nblocks):
	_buf_all(nbits, nblocks),
	_buf_selected(nbits, nblocks)
{
	buffer_all().set_zero();
	buffer_selected().set_zero();
}

PVParallelView::PVHitGraphDataInterface::~PVHitGraphDataInterface()
{
}

void PVParallelView::PVHitGraphDataInterface::shift_left(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_left(n, alpha);
	buffer_selected().shift_zoomed_left(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_right(n, alpha);
	buffer_selected().shift_zoomed_right(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::process_all_buffers(ProcessParams const& params,
                                                          Picviz::PVSelection const& layer_sel,
                                                          Picviz::PVSelection const& sel)
{
	process_buffer_all(params);
	process_buffer_selected(params, sel);
}
