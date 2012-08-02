/**
 * \file PVZonesDrawing.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVZonesDrawing.h>

PVParallelView::PVZonesDrawing::PVZonesDrawing(PVZonesManager & zm, PVBCIDrawingBackend const& backend, PVHSVColor const& colors):
	_zm(zm),
	_draw_backend(&backend),
	_colors(&colors)
{
	_computed_codes = PVBCICode::allocate_codes(NBUCKETS);
}

PVParallelView::PVZonesDrawing::~PVZonesDrawing()
{
	PVBCICode::free_codes(_computed_codes);
}

void PVParallelView::PVZonesDrawing::draw_bci(PVBCIBackendImage& dst_img, uint32_t x_start, size_t width, PVBCICode* codes, size_t n)
{
	_draw_backend->operator()(dst_img, x_start, width, codes, n);
}
