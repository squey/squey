#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVZonesDrawing.h>

namespace PVParallelView {

class PVZonesDrawing;

class PVLinesView
{

	struct ZoneImages
	{
		ZoneImages(PVZonesDrawing const& zd, uint32_t zone_width)
		{
			sel = zd.create_image(zone_width);
			all = zd.create_image(zone_width);
		}

		void set_width(uint32_t zone_width)
		{
			sel->set_width(zone_width);
			all->set_width(zone_width);
		}

		PVBCIBackendImage_p sel;
		PVBCIBackendImage_p all;
	};

public:
	typedef std::vector<ZoneImages> list_zone_images_t;
public:
	PVLinesView(PVZonesDrawing& zones_drawing, PVZoneID nb_drawable_zones, uint32_t zone_width = PVParallelView::ZoneWidth);

public:
	void set_nb_drawable_zones(uint32_t nb_zones) { _nb_drawable_zones = nb_zones;}
	void set_zone_width(uint32_t zone_width) { zone_width = _zone_width; }

public:
	QVector<std::pair<QImage, uint32_t> > translate(int32_t x);
	void render_sel();
	void render_all();

	inline uint32_t get_zone_width(PVZoneID z) const
	{
		return _zd.get_zone_width(z);
	}

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	PVZoneID get_first_zone() const { return _first_zone; }

private:
	PVZonesDrawing& _zd;
	PVZoneID _nb_drawable_zones;
	PVZoneID _first_zone;
	uint32_t _zone_width;

	list_zone_images_t _zones_imgs;
	uint32_t _view_pos;

};

}

#endif
