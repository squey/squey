/**
 * \file PVLinesView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <functional>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvhive/PVCallHelper.h>
#include <picviz/PVSelection.h>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVBCIDrawingBackend;
class PVZonesProcessor;
class PVZoneRenderingBase;
class PVZonesManager;

class PVLinesView
{
	constexpr static size_t bbits = PARALLELVIEW_ZT_BBITS;

private:
	struct ZoneImages
	{
		ZoneImages():
			last_zr_sel(nullptr),
			last_zr_bg(nullptr)
		{ }
	   		   

		ZoneImages(PVBCIDrawingBackend& backend, uint32_t zone_width):
			last_zr_sel(nullptr),
			last_zr_bg(nullptr)
		{
			create_image(backend, zone_width);
		}

		void create_image(PVBCIDrawingBackend& backend, uint32_t zone_width);
		void set_width(uint32_t zone_width);

		void cancel_last_sel();
		void cancel_last_bg();

		void cancel_all_and_wait();

		PVBCIBackendImage_p sel;
		PVBCIBackendImage_p bg;

		PVZoneRenderingBase* last_zr_sel;
		PVZoneRenderingBase* last_zr_bg;
	};

public:
	typedef std::vector<ZoneImages> list_zone_images_t;

public:
	PVLinesView(PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg, QObject* img_update_receiver = NULL, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	PVZoneID get_nb_drawable_zones() const { return _zones_imgs.size(); }
	void set_zone_max_width(uint32_t w);

	int update_number_of_zones(int view_x, uint32_t view_width);

	void cancel_and_wait_all_rendering();

public:
	void render_zone_bg(PVZoneID z, const float zoom_y);
	void render_zone_sel(PVZoneID z, const float zoom_y);
	void render_zone_all_imgs(PVZoneID z, const float zoom_y);

public:
	void translate(int32_t view_x, uint32_t view_width, const float zoom_y);

public:
	void render_all_imgs_bg(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_imgs_sel(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_all_imgs(int32_t view_x, uint32_t view_width, const float zoom_y);

	PVZoneID get_zone_from_scene_pos(int32_t x) const;

	bool set_zone_width(PVZoneID z, uint32_t width);
	//bool set_zone_width_and_render(PVZoneID z, uint32_t width);

	inline const PVZonesManager& get_zones_manager() const { return _zm; }
	inline uint32_t get_zone_width(PVZoneID z) const { assert(z < (PVZoneID) _zones_width.size()); return _zones_width[z]; }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	list_zone_images_t& get_zones_images() { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return picviz_min((PVZoneID)(_first_zone + _zones_imgs.size()-1), get_number_zones()-1); }
	bool is_zone_drawn(PVZoneID z) const { return (z >= get_first_drawn_zone() && z <= get_last_drawn_zone()); }
	PVZoneID get_number_zones() const;
	uint32_t get_zone_absolute_pos(PVZoneID z) const;

	template <class F>
	inline bool set_all_zones_width(F const& f)
	{
		bool has_changed = false;
		for (PVZoneID zid = 0; zid < (PVZoneID) _zones_width.size(); zid++) {
			has_changed |= set_zone_width(zid, f(get_zone_width(zid)));
		}
		return has_changed;
	}

	inline PVBCIDrawingBackend& backend() const { return _backend; }

	inline ZoneImages& get_zone_images(const PVZoneID z)
	{
		return _zones_imgs[get_zone_image_idx(z)];
	}

	PVZoneID get_zone_image_idx(PVZoneID z)
	{
		assert(is_zone_drawn(z));
		return z-get_first_drawn_zone();
	}

private:
	PVZoneID get_image_index_of_zone(PVZoneID z) const;

	inline void update_zone_sel_img_width(PVZoneID z)
	{
		get_zone_images(z).sel->set_width(get_zone_width(z));
	}

	inline void update_zone_bg_img_width(PVZoneID z)
	{
		get_zone_images(z).bg->set_width(get_zone_width(z));
	}
	
	void visit_all_zones_to_render(uint32_t view_width, std::function<void(PVZoneID)> const& fzone);

	PVZoneID set_new_view(int32_t new_view_x, uint32_t view_width)
	{
		// Change view_x, and set new first zone
		// Returns the old first zone
		_visible_view_x = new_view_x;
		PVZoneID new_first_zone = get_first_zone_from_viewport(new_view_x, view_width);

		PVZoneID pre_first_zone = _first_zone;

		PVLOG_INFO("set_new_view: new=%d/old=%d\n", new_first_zone, pre_first_zone);

		_first_zone = new_first_zone;
		return pre_first_zone;
	}

	void do_translate(PVZoneID pre_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw);

	PVZoneID get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const;

	void left_shift_images(PVZoneID s);
	void right_shift_images(PVZoneID s);

	void connect_zr(PVZoneRenderingBase* zr, const char* slot);
	void call_refresh_slots(int zid);

private:
	PVZoneID _first_zone;
	uint32_t _zone_max_width;
	int32_t _visible_view_x;

	std::vector<uint32_t> _zones_width;

	list_zone_images_t _zones_imgs;

	PVZonesProcessor& _processor_sel;
	PVZonesProcessor& _processor_bg;

	PVZonesManager const& _zm;
	PVBCIDrawingBackend& _backend;

	QObject* _img_update_receiver; 
};

}

#endif
