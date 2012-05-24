#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvkernel/core/PVAlgorithms.h>

namespace PVParallelView {

class PVZonesDrawing;

class PVLinesView
{
	struct ZoneImages
	{
		ZoneImages() { }

		ZoneImages(PVZonesDrawing const& zd, uint32_t zone_width)
		{
			create_image(zd, zone_width);
		}

		void set_width(uint32_t zone_width)
		{
			sel->set_width(zone_width);
			bg->set_width(zone_width);
		}

		void create_image(PVZonesDrawing const& zd, uint32_t zone_width)
		{
			sel = zd.create_image(zone_width);
			bg = zd.create_image(zone_width);
		}

		PVBCIBackendImage_p sel;
		PVBCIBackendImage_p bg;
	};

public:
	typedef std::vector<ZoneImages> list_zone_images_t;

public:
	PVLinesView(PVZonesDrawing& zones_drawing, PVZoneID nb_drawable_zones, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	void set_zone_max_width(uint32_t w);

public:
	void translate(int32_t view_x, uint32_t view_width);
	void render_all(int32_t view_x, uint32_t view_width)
	{
		_first_zone = get_first_zone_from_viewport(view_x, view_width);
		_visibile_view_x = view_x;
		render_all_imgs(view_width);
	}
	void render_bg(uint32_t view_width) const;
	void render_sel(uint32_t view_width) const;
	void render_all_imgs(uint32_t view_width) const;

	inline PVZoneID get_zone_from_scene_pos(int32_t x) const { return get_zones_manager().get_zone_id(x); }

	bool set_zone_width_and_render(PVZoneID z, uint32_t width);

	inline const PVZonesDrawing& get_zones_drawing() const { return _zd; }
	inline const PVZonesManager& get_zones_manager() const { return _zd.get_zones_manager(); }
	inline uint32_t get_zone_width(PVZoneID z) const { return _zd.get_zone_width(z); }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return _first_zone + _zones_imgs.size(); }
	bool is_zone_drawn(PVZoneID z) const { return (z >= get_first_drawn_zone() && z <= get_last_drawn_zone()); }
	inline uint32_t get_zone_absolute_pos(PVZoneID z) const { return get_zones_manager().get_zone_absolute_pos(z); }

private:
	void render_zone_all_imgs(PVZoneID z, ZoneImages const& zi) const;

	PVZoneID get_image_index_of_zone(PVZoneID z) const;
	
	template <class F>
	void render_all_zones(uint32_t view_width, F const& fzone) const
	{
		int32_t view_x = _visibile_view_x;
		if (view_x < 0) {
			uint32_t unused_width = (uint32_t) (-view_x);
			if (unused_width >= view_width) {
				unused_width = view_width;
			}
			view_width -= unused_width;
			view_x = 0;
		}

		PVZoneID left_invisible_zone;
		PVZoneID right_invisible_zone;

		const PVZoneID nzones_total = get_zones_manager().get_number_zones();
		const PVZoneID zfirst_visible = get_zone_from_scene_pos(view_x);
		PVZoneID zones_to_draw = _zones_imgs.size();
		assert(zfirst_visible >= _first_zone);
		assert(zfirst_visible < _first_zone+zones_to_draw);

		left_invisible_zone = zfirst_visible;

		// Process visible zones
		uint32_t cur_width = 0;
		PVZoneID cur_z = zfirst_visible;
		while (cur_width < view_width && cur_z < nzones_total) {
			fzone(cur_z);
			const uint32_t offset = get_zone_width(cur_z) + PVParallelView::AxisWidth;
			cur_width += offset;
			cur_z++;
			zones_to_draw--;
		}
		right_invisible_zone = cur_z;

		while (zones_to_draw > 0) {
			if (left_invisible_zone > _first_zone) {
				left_invisible_zone--;
				assert(left_invisible_zone >= _first_zone);
				fzone(left_invisible_zone);
				zones_to_draw--;
				if (zones_to_draw == 0) {
					break;
				}
			}
			if (right_invisible_zone < nzones_total) {
				fzone(right_invisible_zone);
				right_invisible_zone++;
				zones_to_draw--;
			}
		}
	}

	template <class F>
	void do_translate(int32_t view_x, uint32_t view_width, F const& fzone_draw)
	{
		int32_t new_view_x = view_x;
		_visibile_view_x = new_view_x;
		PVZoneID new_first_zone = get_first_zone_from_viewport(view_x, view_width);
		if (new_first_zone == _first_zone) {
			// "Le changement, c'est pas maintenant !"
			PVLOG_INFO("(do_translate) same first zone. Do nothing.\n");
			return;
		}

		const PVZoneID nzones_img = _zones_imgs.size();
		const PVZoneID diff = std::abs(new_first_zone - _first_zone);
		if (diff >= nzones_img) {
			_first_zone = new_first_zone;
			render_all_zones(view_width, fzone_draw);
			return;
		}
		PVLOG_INFO("(do translate) first zone: %d\n", new_first_zone);

		if (new_first_zone > _first_zone) {
			const PVZoneID n = diff;
			left_shift_images(n);
			const PVZoneID nimgs = _zones_imgs.size();
			for (PVZoneID z = nimgs-n; z < nimgs; z++) {
				fzone_draw(z+new_first_zone);
			}
		}
		else {
			const PVZoneID n = diff;
			right_shift_images(diff);
			for (PVZoneID z = 0; z < n; z++) {
				fzone_draw(z+new_first_zone);
			}
		}
		_first_zone = new_first_zone;
	}

	PVZoneID get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const;

	void left_shift_images(PVZoneID s);
	void right_shift_images(PVZoneID s);

private:
	PVZonesDrawing& _zd;
	PVZoneID _first_zone;
	uint32_t _zone_max_width;
	int32_t _visibile_view_x;

	list_zone_images_t _zones_imgs;
};

}

#endif
