#ifndef PVPARALLELVIEW_PVZONESDRAWING_H
#define PVPARALLELVIEW_PVZONESDRAWING_H

namespace PVParallelView {

class PVZonesManager;
class PVBCIDrawingBackend;

class PVZonesDrawing
{
public:
	PVZonesDrawing(PVZonesManager const& zm):
		_pos(0),
		_zoom_x(1)
	{ }

public:
	template <typename Tree, typename Fbci>
	void draw(QImage& dst_img, Tree const& tree, Fbci const& f_bci);

private:
	template <typename Tree, typename Fbci>
	void draw_zone(QImage& dst_img, PVZoneID zone, Tree const& tree, Fbci const& f_bci);

pivate:
	int32_t _pos;
	int32_t _zoom_x;

	PVZonesManager const& _zm;
};

}

#endif
