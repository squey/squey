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

void PVParallelView::PVZonesDrawing::draw_bci(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, PVBCICode* codes, size_t n)
{
	uint32_t width = _zm.get_zone_width(zone);
	dst_img.set_width(width);
	_draw_backend->operator()(dst_img, x_start, width, codes, n);
}
