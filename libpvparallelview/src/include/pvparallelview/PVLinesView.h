/**
 * \file PVLinesView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <functional>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <picviz/PVSelection.h>

#include <pvhive/PVCallHelper.h>

#include <boost/integer/static_log2.hpp>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVBCIDrawingBackend;
class PVZonesProcessor;
class PVZonesManager;

class PVLinesView
{
	constexpr static size_t bbits = PARALLELVIEW_ZT_BBITS;

private:
	struct SingleZoneImages
	{
		PVBCIBackendImage_p sel;
		PVBCIBackendImage_p bg;

		PVZoneRenderingBase_p last_zr_sel;
		PVZoneRenderingBase_p last_zr_bg;

		SingleZoneImages():
			last_zr_sel(),
			last_zr_bg()
		{ }
	   		   

		SingleZoneImages(PVBCIDrawingBackend& backend, uint32_t zone_width):
			last_zr_sel(),
			last_zr_bg()
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
			_base_width = PVParallelView::ZoneBaseWidth;
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

		uint32_t get_width() const;
		
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
	inline PVBCIDrawingBackend& backend() const { return _backend; }

	void cancel_and_wait_all_rendering();
	
	void decrease_base_zoom_level_of_zone(PVZoneID zone_id);
	void decrease_global_zoom_level();

	inline PVZoneID get_first_visible_zone_index() const { return _first_zone; }
	inline PVZoneID get_last_visible_zone_index() const { return picviz_min((PVZoneID)(_first_zone + get_number_of_visible_zones()-1), get_number_of_managed_zones()-1); }
	uint32_t get_left_border_position_of_zone_in_scene(PVZoneID zone_id) const;

	PVZoneID get_number_of_managed_zones() const;
	PVZoneID get_number_of_visible_zones() const { return _list_of_single_zone_images.size(); }

	int32_t get_left_border_of_scene() const { return get_left_border_position_of_zone_in_scene(0); }
	int32_t get_right_border_of_scene() const
	{
		const PVZoneID last_z = get_number_of_managed_zones()-1;
		return get_left_border_position_of_zone_in_scene(last_z) + 2*PVParallelView::AxisWidth + get_zone_width(last_z);
	}

	inline SingleZoneImages& get_single_zone_images(const PVZoneID zone_id) { return _list_of_single_zone_images[get_zone_index_offset(zone_id)]; }

	PVZoneID get_zone_from_scene_pos(int32_t x) const;
	PVZoneID get_zone_index_offset(PVZoneID zone_id) { assert(is_zone_drawn(zone_id)); return zone_id-get_first_visible_zone_index(); }
	
	inline const PVZonesManager& get_zones_manager() const { return _zm; }
//	inline uint32_t get_zone_width(PVZoneID zone_id) const { assert(zone_id < (PVZoneID) _zones_width.size()); return _zones_width[zone_id]; }
	uint32_t get_zone_width(PVZoneID zone_id) const;
		
	const list_zone_images_t& get_zones_images() const { return _list_of_single_zone_images; }
	list_zone_images_t& get_zones_images() { return _list_of_single_zone_images; }

	void increase_base_zoom_level_of_zone(PVZoneID zone_id);
	void increase_global_zoom_level();

	/**
	 * Initialize the zones width for make them get in the viewport. If there not enough
	 * space, the default zone width is used to make zones understandable.
	 *
	 * @param view_width the width of the view in which the zones should fit in
	 *
	 * @return true if the zones fit in width; otherwise false
	 */
	bool initialize_zones_width(int view_width);

	bool is_zone_drawn(PVZoneID zone_id) const { return (zone_id >= get_first_visible_zone_index() && zone_id <= get_last_visible_zone_index()); }

	void render_all_zones_bg_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_images(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_all_zones_sel_image(int32_t view_x, uint32_t view_width, const float zoom_y);
	void render_single_zone_bg_image(PVZoneID zone_id, const float zoom_y);
	void render_single_zone_images(PVZoneID zone_id, const float zoom_y);
	void render_single_zone_sel_image(PVZoneID zone_id, const float zoom_y);

	void set_nb_drawable_zones(PVZoneID nb_zones);

	void set_zone_max_width(uint32_t w);
	bool set_zone_width(PVZoneID zone_id, uint32_t width);
	//bool set_zone_width_and_render(PVZoneID zone_id, uint32_t width);

	void translate(int32_t view_x, uint32_t view_width, const float zoom_y);

	int update_number_of_zones(int view_x, uint32_t view_width);


public:
	template <class F>
	inline bool set_all_zones_width(F const& f)
	{
		bool has_changed = false;
		for (PVZoneID zone_id = 0; zone_id < (PVZoneID) _zones_width.size(); zone_id++) {
			has_changed |= set_zone_width(zone_id, f(get_zone_width(zone_id)));
		}
		return has_changed;
	}




private:
	PVZoneID get_image_index_of_zone(PVZoneID zone_id) const;

	inline void update_zone_sel_img_width(PVZoneID zone_id)
	{
		get_single_zone_images(zone_id).sel->set_width(get_zone_width(zone_id));
	}

	inline void update_zone_bg_img_width(PVZoneID zone_id)
	{
		get_single_zone_images(zone_id).bg->set_width(get_zone_width(zone_id));
	}
	
	void visit_all_zones_to_render(uint32_t view_width, std::function<void(PVZoneID)> const& fzone);

	PVZoneID set_new_view(int32_t new_view_x, uint32_t view_width)
	{
		// Change view_x
		_visible_view_x = new_view_x;
		
		// and set new first zone
		PVZoneID previous_first_zone = _first_zone;
		_first_zone = update_and_get_first_zone_from_viewport(new_view_x, view_width);

		// Returns the previous first zone index
		return previous_first_zone;
	}

	void do_translate(PVZoneID previous_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw);

	PVZoneID update_and_get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const;

	void left_rotate_single_zone_images(PVZoneID s);
	void right_rotate_single_zone_images(PVZoneID s);

	void connect_zr(PVZoneRenderingBase* zr, const char* slot);
	void call_refresh_slots(PVZoneID zone_id);

	
private:
	PVBCIDrawingBackend& _backend;

	PVZoneID _first_zone;
	
	QObject* _img_update_receiver;

	list_zone_images_t _list_of_single_zone_images;
	list_zone_width_with_zoom_level_t _list_of_zone_width_with_zoom_level;

	PVZonesProcessor& _processor_sel;
	PVZonesProcessor& _processor_bg;

	
	int32_t _visible_view_x;

	PVZonesManager const& _zm;
	uint32_t _zone_max_width;
	
	std::vector<uint32_t> _zones_width;

};

}

#endif
