#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <pvparallelview/PVBCIBackendImage_types.h>

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
	PVLinesView(PVZonesDrawing& zones_drawing, uint32_t nb_zones, uint32_t size_zone);

public:
	void set_nb_zones(uint32_t nb_zones);
	void set_zone_width(uint32_t zone_width);

public:
	QVector<std::pair<QImage, uint32_t> > translate(int32_t x);

private:
	PVZonesDrawing& _zd;
	std::vector<ZoneImages> _zones_imgs;
	uint32_t _view_pos;

};

}

#endif
