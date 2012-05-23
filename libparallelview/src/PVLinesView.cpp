#include <pvparallelview/PVLinesView.h>

PVParallelView::PVLinesView::PVLinesView(PVZonesDrawing& zones_drawing, PVZoneID nb_zones, uint32_t zone_width /* = PVParallelView::ZoneWidth */) :
	_zd(zones_drawing),
	_nb_drawable_zones(nb_zones),
	_zone_width(zone_width),
	_first_zone(0)
{
	_zones_imgs.reserve(nb_zones);
	for (int i = 0; i < _nb_drawable_zones; i++) {
		_zones_imgs.push_back(ZoneImages(_zd, zone_width));
	}
}

QVector<std::pair<QImage, uint32_t> > PVParallelView::PVLinesView::translate(int32_t x)
{
}

void PVParallelView::PVLinesView::render_sel()
{
	for (PVZoneID i = 0; i < _nb_drawable_zones; i++) {
		//_zd.draw_zone<PVParallelView::PVZoneTree>(_zones_imgs[i].all, 0, i+_first_zone, &PVParallelView::PVZoneTree::browse_tree_sel_bci);
	}
}

void PVParallelView::PVLinesView::render_all()
{
	for (PVZoneID i = 0; i < _nb_drawable_zones; i++) {
		_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[i].all, 0, i+_first_zone, &PVParallelView::PVZoneTree::browse_tree_bci);
	}
}
