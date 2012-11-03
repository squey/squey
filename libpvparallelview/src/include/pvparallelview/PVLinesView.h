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
	struct SingleZoneImages
	{
		PVBCIBackendImage_p sel;
		PVBCIBackendImage_p bg;

		PVZoneRenderingBase* last_zr_sel;
		PVZoneRenderingBase* last_zr_bg;

		SingleZoneImages():
			last_zr_sel(nullptr),
			last_zr_bg(nullptr)
		{ }
	   		   

		SingleZoneImages(PVBCIDrawingBackend& backend, uint32_t zone_width):
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
	};
	
	struct ZoneWidthWithZoomLevel
	{
		ZoneWidthWithZoomLevel()
		{
			_base_width = 128;
			_base_zoom_level = 0;
		}
		
		ZoneWidthWithZoomLevel(int16_t base_width, int16_t base_zoom_level)
		{
			_base_width = base_width;
			_base_zoom_level = base_zoom_level;
		}
		
		void decrease_zoom_level();
		
		int16_t get_base_zoom_level();
		int16_t get_base_width();
		
		void increase_zoom_level();
		
		void set_base_width(int16_t base_width);
		void set_base_zoom_level(int16_t base_zoom_level);
		
		int16_t _base_width;
		int16_t _base_zoom_level;
	};

public:
	typedef std::vector<SingleZoneImages> list_zone_images_t;
	typedef std::vector<ZoneWidthWithZoomLevel> list_zone_width_with_zoom_level_t;

public:
	PVLinesView(PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg, QObject* img_update_receiver = NULL, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	PVZoneID get_nb_drawable_zones() const { return _zones_imgs.size(); }
	void set_zone_max_width(uint32_t w);

	int update_number_of_zones(int view_x, uint32_t view_width);

	void cancel_and_wait_all_rendering();

public:
	void render_all_zones_bg_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_images(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_sel_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_single_zone_bg_image(PVZoneID zone_id, const float zoom_y);
	void render_single_zone_images(PVZoneID zone_id, const float zoom_y);
	void render_single_zone_sel_image(PVZoneID zone_id, const float zoom_y);

public:
	void translate(int32_t view_x, uint32_t view_width, const float zoom_y);

public:
	PVZoneID get_zone_from_scene_pos(int32_t x) const;

	bool set_zone_width(PVZoneID zone_id, uint32_t width);
	//bool set_zone_width_and_render(PVZoneID zone_id, uint32_t width);

	inline const PVZonesManager& get_zones_manager() const { return _zm; }
	inline uint32_t get_zone_width(PVZoneID zone_id) const { assert(zone_id < (PVZoneID) _zones_width.size()); return _zones_width[zone_id]; }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	list_zone_images_t& get_zones_images() { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return picviz_min((PVZoneID)(_first_zone + _zones_imgs.size()-1), get_number_of_zones()-1); }
	bool is_zone_drawn(PVZoneID zone_id) const { return (zone_id >= get_first_drawn_zone() && zone_id <= get_last_drawn_zone()); }
	uint32_t get_zone_absolute_pos(PVZoneID zone_id) const;
	PVZoneID get_number_of_zones() const;

	template <class F>
	inline bool set_all_zones_width(F const& f)
	{
		bool has_changed = false;
		for (PVZoneID zone_id = 0; zone_id < (PVZoneID) _zones_width.size(); zone_id++) {
			has_changed |= set_zone_width(zone_id, f(get_zone_width(zone_id)));
		}
		return has_changed;
	}

	inline PVBCIDrawingBackend& backend() const { return _backend; }

	inline SingleZoneImages& get_single_zone_images(const PVZoneID zone_id)
	{
		return _zones_imgs[get_zone_image_idx(zone_id)];
	}

	PVZoneID get_zone_image_idx(PVZoneID zone_id)
	{
		assert(is_zone_drawn(zone_id));
		return zone_id-get_first_drawn_zone();
	}

private:
	PVZoneID get_image_index_of_zone(PVZoneID zone_id) const;

	// FIXME : not used ??
// 	inline void update_zone_sel_img_width(PVZoneID zone_id)
// 	{
// 		get_single_zone_images(zone_id).sel->set_width(get_zone_width(zone_id));
// 	}

	// FIXME : not used ??
// 	inline void update_zone_bg_img_width(PVZoneID zone_id)
// 	{
// 		get_single_zone_images(zone_id).bg->set_width(get_zone_width(zone_id));
// 	}
	
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
	void call_refresh_slots(PVZoneID zone_id);

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
