//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
