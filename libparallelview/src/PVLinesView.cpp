#include <pvparallelview/PVLinesView.h>

PVParallelView::PVLinesView::PVLinesView(PVZonesDrawing& zones_drawing, uint32_t nb_zones, uint32_t zone_width /* = PVParallelView::ZoneWidth */) :
	_zd(zones_drawing),
	_nb_zones(nb_zones),
	_zone_width(zone_width)
{

}

QVector<std::pair<QImage, uint32_t> > PVParallelView::PVLinesView::translate(int32_t x)
{
}

void PVParallelView::PVLinesView::render_sel()
{
}

void PVParallelView::PVLinesView::render_all()
{
}
