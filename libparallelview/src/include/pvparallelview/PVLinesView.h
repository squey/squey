/**
 * \file PVLinesView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVRenderingJob.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvkernel/core/PVAlgorithms.h>

#include <QFuture>

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

template <size_t Bbits>
class PVZonesDrawing;

class PVLinesView
{
	constexpr static size_t bbits = NBITS_INDEX;

public:
	typedef PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

private:
	struct ZoneImages
	{
		ZoneImages() { }

		ZoneImages(zones_drawing_t const& zd, uint32_t zone_width)
		{
			create_image(zd, zone_width);
		}

		void set_width(uint32_t zone_width)
		{
			sel->set_width(zone_width);
			bg->set_width(zone_width);
		}

		void create_image(zones_drawing_t const& zd, uint32_t zone_width)
		{
			sel = zd.create_image(zone_width);
			bg = zd.create_image(zone_width);
		}

		backend_image_p_t sel;
		backend_image_p_t bg;
	};

public:
	typedef std::vector<ZoneImages> list_zone_images_t;

public:
	PVLinesView(zones_drawing_t& zones_drawing, PVZoneID nb_drawable_zones, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	void set_zone_max_width(uint32_t w);

public:
	void translate(int32_t view_x, uint32_t view_width);
	QFuture<void> translate(int32_t view_x, uint32_t view_width, PVRenderingJob& job);

	void render_all(int32_t view_x, uint32_t view_width);
	QFuture<void> render_all(int32_t view_x, uint32_t view_width, PVRenderingJob& job);

	void render_all_imgs(uint32_t view_width);
	QFuture<void> render_all_imgs(uint32_t view_width, PVRenderingJob& job);

	void render_bg(uint32_t view_width);

	void render_sel(uint32_t view_width);
	QFuture<void> render_sel(uint32_t view_width, PVRenderingJob& job);

	void update_sel_from_zone(uint32_t view_width, PVZoneID zid, const Picviz::PVSelection& sel);
	QFuture<void> update_sel_from_zone(uint32_t view_width, PVZoneID zid_sel, const Picviz::PVSelection& sel, PVRenderingJob& job);

	void render_zone_all_imgs(PVZoneID z);
	QFuture<void> render_zone_all_imgs(PVZoneID z, PVRenderingJob& job);

	inline PVZoneID get_zone_from_scene_pos(int32_t x) const { return get_zones_manager().get_zone_id(x); }

	inline bool set_zone_width(PVZoneID z, uint32_t width)
	{
		// Returns true if width was actual changed
		uint32_t old_width = get_zone_width(z);
		get_zones_manager().set_zone_width(z, width);
		return get_zone_width(z) != old_width;
	}
	bool set_zone_width_and_render(PVZoneID z, uint32_t width);

	inline const zones_drawing_t& get_zones_drawing() const { return _zd; }
	inline const PVZonesManager& get_zones_manager() const { return _zd.get_zones_manager(); }
	inline PVZonesManager& get_zones_manager() { return _zd.get_zones_manager(); }
	inline uint32_t get_zone_width(PVZoneID z) const { return _zd.get_zone_width(z); }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	list_zone_images_t& get_zones_images() { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return picviz_min((PVZoneID)(_first_zone + _zones_imgs.size()-1), (PVZoneID)get_zones_manager().get_number_zones()-1); }
	bool is_zone_drawn(PVZoneID z) const { return (z >= get_first_drawn_zone() && z <= get_last_drawn_zone()); }
	inline uint32_t get_zone_absolute_pos(PVZoneID z) const { return get_zones_manager().get_zone_absolute_pos(z); }

	template <class F>
	inline void set_all_zones_width(F const& f) { get_zones_manager().set_zones_width(f); }

	template <class F>
	bool set_all_zones_width_and_render(int32_t visible_view_x, uint32_t width, F const& f)
	{
		//width = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
		get_zones_manager().set_zones_width(f);
		render_all(visible_view_x, width);
		return true;
	}

private:
	PVZoneID get_image_index_of_zone(PVZoneID z) const;

	inline void update_zone_images_width(PVZoneID z)
	{
		assert(is_zone_drawn(z));
		_zones_imgs[z-get_first_drawn_zone()].set_width(get_zone_width(z));
	}
	
	template <class F>
	void render_all_zones(uint32_t view_width, F const& fzone, PVRenderingJob* job = NULL)
	{
		int32_t view_x = _visible_view_x;
		if (view_x < 0) {
			// We start to be too far on the left...
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
		while (cur_width < view_width && cur_z < nzones_total && zones_to_draw > 0) {
			if (job && job->should_cancel()) {
				return;
			}
			fzone(cur_z);
			if (job) {
				job->zone_finished(cur_z);
			}
			const uint32_t offset = get_zone_width(cur_z) + PVParallelView::AxisWidth;
			cur_width += offset;
			cur_z++;
			zones_to_draw--;
		}
		right_invisible_zone = cur_z;

		// Process hidden zones
		while (zones_to_draw > 0) {
			bool one_done = false;
			if (left_invisible_zone > _first_zone) {
				left_invisible_zone--;
				assert(left_invisible_zone >= _first_zone);
				if (job && job->should_cancel()) {
					return;
				}
				fzone(left_invisible_zone);
				if (job) {
					job->zone_finished(left_invisible_zone);
				}
				zones_to_draw--;
				if (zones_to_draw == 0) {
					break;
				}
				one_done = true;
			}
			if (right_invisible_zone < nzones_total) {
				if (job && job->should_cancel()) {
					return;
				}
				fzone(right_invisible_zone);
				if (job) {
					job->zone_finished(right_invisible_zone);
				}
				right_invisible_zone++;
				zones_to_draw--;
				one_done = true;
			}
			if (!one_done) {
				break;
			}
		}
	}

	PVZoneID set_new_view(int32_t new_view_x, uint32_t view_width)
	{
		// Change view_x, and set new first zone
		// Returns the old first zone
		_visible_view_x = new_view_x;
		PVZoneID new_first_zone = get_first_zone_from_viewport(new_view_x, view_width);

		PVZoneID pre_first_zone = _first_zone;

		_first_zone = new_first_zone;
		return pre_first_zone;
	}

	template <class F>
	void do_translate(PVZoneID pre_first_zone, uint32_t view_width, F const& fzone_draw, PVRenderingJob* job = NULL)
	{
		const PVZoneID nzones_img = _zones_imgs.size();
		const PVZoneID diff = std::abs(_first_zone - pre_first_zone);
		if (diff >= nzones_img) {
			render_all_zones(view_width, fzone_draw, job);
			return;
		}
		PVLOG_INFO("(do translate) first zone: %d\n", _first_zone);

		if (_first_zone > pre_first_zone) {
			// Translation to the left 
			
			const PVZoneID n = diff;
			left_shift_images(n);

			const PVZoneID nimgs = _zones_imgs.size();
			PVZoneID first_z_to_render = _first_zone + nimgs - n;
			const PVZoneID last_z = picviz_min(nimgs+_first_zone, get_zones_manager().get_number_zones());

			// If a rendering job is provided, tell him that we virtually have rendered from _first_zone to first_z_to_render images
			if (job) {
				for (PVZoneID z = _first_zone; z < first_z_to_render; z++) {
					job->zone_finished(z);
				}
			}

			for (PVZoneID z = first_z_to_render; z < last_z; z++) {
				if (job && job->should_cancel()) {
					return;
				}
				fzone_draw(z);
				if (job) {
					job->zone_finished(z);
				}
			}
		}
		else {
			// Translation to the right

			right_shift_images(diff);
			const PVZoneID n = diff;
			PVZoneID first_z_to_render = _first_zone;
			const PVZoneID last_z = picviz_min(_first_zone + n, get_zones_manager().get_number_zones());

			// If a rendering job is provided, tell him that we virtually have rendered from last_z to get_last_drawn_zone()
			if (job) {
				for (PVZoneID z = last_z; z <= get_last_drawn_zone(); z++) {
					job->zone_finished(z);
				}
			}

			for (PVZoneID z = last_z-1; z >= first_z_to_render; z--) {
				if (job && job->should_cancel()) {
					return;
				}
				fzone_draw(z);
				if (job) {
					job->zone_finished(z);
				}
			}
		}
	}

	PVZoneID get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const;

	void left_shift_images(PVZoneID s);
	void right_shift_images(PVZoneID s);

private:
	zones_drawing_t& _zd;
	PVZoneID _first_zone;
	uint32_t _zone_max_width;
	int32_t _visible_view_x;

	list_zone_images_t _zones_imgs;
};

}

#endif
