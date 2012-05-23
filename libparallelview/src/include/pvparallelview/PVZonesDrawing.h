#ifndef PVPARALLELVIEW_PVZONESDRAWING_H
#define PVPARALLELVIEW_PVZONESDRAWING_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVBCIBackendImage.h>

#include <boost/utility.hpp>

#include <cassert>

namespace PVParallelView {

class PVBCIDrawingBackend;
class PVHSVColor;

class PVZonesDrawing: public boost::noncopyable
{
public:
	PVZonesDrawing(PVZonesManager const& zm, PVBCIDrawingBackend const& backend, PVHSVColor const& colors);
	~PVZonesDrawing();

/*public:
	template <typename Tree, typename Fbci>
	void draw(QImage& dst_img, Tree const& tree, Fbci const& f_bci);*/

public:
	inline void set_backend(PVBCIDrawingBackend const& backend) { _draw_backend = &backend; }

public:
	inline PVBCIBackendImage_p create_image(size_t width) const { assert(_draw_backend); return _draw_backend->create_image(width); }

public:
	template <class Tree, class Fbci>
	void draw_zones(PVBCIBackendImage* dst_imgs, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			draw_zone<Tree, Fbci>(dst_imgs[zone_start], 0, zone, f_bci);
		}
	}

	template <class Tree, class Fbci>
	uint32_t draw_zones(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			assert(x_start + _zm.get_zone_width(zone) + AxisWidth <= dst_img.width());
			draw_zone<Tree,Fbci>(dst_img, x_start, zone, f_bci);
			x_start += _zm.get_zone_width(zone) + AxisWidth;
		}
		return x_start;
	}

	template <class Tree, class Fbci>
	void draw_zone(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, Fbci const& f_bci)
	{
		Tree const& zone_tree = _zm.get_zone_tree<Tree>(zone);
		PVLOG_INFO("draw_zone: tree pointer: %p\n", &zone_tree);
		size_t ncodes = (zone_tree.*f_bci)(_colors, _computed_codes);
		draw_bci(dst_img, x_start, zone, _computed_codes, ncodes);
	}

	inline uint32_t get_zone_width(PVZoneID z) const
	{
		return _zm.get_zone_width(z);
	}

	inline const PVZonesManager&  get_zones_manager() const
	{
		return _zm;
	}

private:
	void draw_bci(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, PVBCICode* codes, size_t n);

private:
	PVZonesManager const& _zm;
	PVBCIDrawingBackend const* _draw_backend;
	PVHSVColor const* _colors;
	PVBCICode* _computed_codes;
};

}

#endif
