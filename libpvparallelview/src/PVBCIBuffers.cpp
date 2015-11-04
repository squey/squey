/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVBCIBuffers.h>
#include <pvparallelview/PVBCIDrawingBackend.h>

#include <pvparallelview/PVZonesDrawing.h>

// Buffers for 10 and 11-bits BCI codes
PVParallelView::PVBCIBuffers<BCI_BUFFERS_COUNT> PVParallelView::__impl::PVZonesDrawingBase::_computed_codes;

PVParallelView::PVBCIBuffersAlloc::bci_base_type* PVParallelView::PVBCIBuffersAlloc::allocate(size_t n, PVBCIDrawingBackend& backend)
{
	return backend.allocate_bci(n);
}

void PVParallelView::PVBCIBuffersAlloc::free(bci_base_type* buf, PVBCIDrawingBackend& backend)
{
	backend.free_bci(buf);
}
