/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVHitGraphDataInterface.h>

//
// PVHitGraphData
//

PVParallelView::PVHitGraphDataInterface::PVHitGraphDataInterface(uint32_t nbits, uint32_t nblocks)
    : _buf_all(nbits, nblocks), _buf_selected(nbits, nblocks), _buf_selectable(nbits, nblocks)
{
	buffer_all().set_zero();
	buffer_selected().set_zero();
	buffer_selectable().set_zero();
}

PVParallelView::PVHitGraphDataInterface::~PVHitGraphDataInterface() = default;

void PVParallelView::PVHitGraphDataInterface::shift_left(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_left(n, alpha);
	buffer_selected().shift_zoomed_left(n, alpha);
	buffer_selectable().shift_zoomed_left(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::shift_right(const uint32_t n, const double alpha)
{
	buffer_all().shift_zoomed_right(n, alpha);
	buffer_selected().shift_zoomed_right(n, alpha);
	buffer_selectable().shift_zoomed_right(n, alpha);
}

void PVParallelView::PVHitGraphDataInterface::process_all_buffers(
    ProcessParams const& params,
    Inendi::PVSelection const& layer_sel,
    Inendi::PVSelection const& sel)
{
	process_buffer_all(params);
	process_buffer_selected(params, sel);
	process_buffer_selectable(params, layer_sel);
}
