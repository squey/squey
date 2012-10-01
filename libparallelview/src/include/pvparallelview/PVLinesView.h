/**
 * \file PVLinesView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVLINESVIEW_H
#define PVPARALLELVIEW_PVLINESVIEW_H

#include <functional>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVRenderingJob.h>
#include <pvparallelview/PVZonesDrawing.h>
#include <pvkernel/core/PVAlgorithms.h>
#include <pvhive/PVCallHelper.h>
#include <picviz/PVSelection.h>

#include <QFuture>

namespace tbb {
class task;
class task_group;
}

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

template <size_t Bbits>
class PVZonesDrawing;

class PVLinesView
{
	constexpr static size_t bbits = PARALLELVIEW_ZT_BBITS;

public:
	typedef PVZonesDrawing<bbits> zones_drawing_t;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;
	typedef typename zones_drawing_t::render_group_t render_group_t;
	typedef typename zones_drawing_t::bci_backend_t bci_backend_t;

private:
	struct ZoneImages
	{
		ZoneImages() { }

		ZoneImages(PVParallelView::PVLinesView::zones_drawing_t* zd, uint32_t zone_width)
		{
			create_image(zd, zone_width);
		}

		void set_width(uint32_t zone_width)
		{
			sel->set_width(zone_width);
			bg->set_width(zone_width);
		}

		void create_image(PVParallelView::PVLinesView::zones_drawing_t* zd, uint32_t zone_width)
		{
			sel = zd->create_image(zone_width);
			bg = zd->create_image(zone_width);
		}

		backend_image_p_t sel;
		backend_image_p_t bg;
	};

public:
	typedef std::vector<ZoneImages> list_zone_images_t;

public:
	PVLinesView(zones_drawing_t& zd, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	PVZoneID get_nb_drawable_zones() const { return _zones_imgs.size(); }
	void set_zone_max_width(uint32_t w);

	int update_number_of_zones(int view_x, uint32_t view_width);

public:
	void render_all_imgs_bg(uint32_t view_width, tbb::task_group& grp_bg, const float zoom_y, PVRenderingJob* job);

	void update_sel_tree(uint32_t view_width, const Picviz::PVSelection& sel, tbb::task* root);

	void render_zone_all_imgs(PVZoneID z, const Picviz::PVSelection& sel, tbb::task_group& grp_bg, tbb::task* root_sel, const float zoom_y, PVRenderingJob* job);
	void render_all_zones_all_imgs(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, tbb::task_group& grp_bg, tbb::task* root_sel, const float zoom_y, PVRenderingJob* job_bg);

	void render_zone_bg(PVZoneID z, const float zoom_y, PVRenderingJob* job);
	void render_zone_sel(PVZoneID z, const float zoom_y, PVRenderingJob* job);

	void translate(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, tbb::task* root_sel, tbb::task_group& grp_bg, const float zoom_y, PVRenderingJob* job);

	void cancel_sel_rendering()
	{
		get_zones_drawing()->cancel_group(_render_grp_sel);
	}

	void cancel_all_rendering()
	{
		cancel_sel_rendering();
		get_zones_drawing()->cancel_group(_render_grp_bg);
	}

	PVZoneID get_zone_from_scene_pos(int32_t x) const;

	inline bool set_zone_width(PVZoneID z, uint32_t width)
	{
		assert(z < (PVZoneID) _zones_width.size());
		// Returns true if width was actual changed
		uint32_t old_width = get_zone_width(z);
		_zones_width[z] = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
		return get_zone_width(z) != old_width;
	}
	//bool set_zone_width_and_render(PVZoneID z, uint32_t width);

	inline zones_drawing_t* get_zones_drawing() { return _zd; }
	inline const PVZonesManager& get_zones_manager() const { return _zd->get_zones_manager(); }
	inline PVZonesManager& get_zones_manager() { return _zd->get_zones_manager(); }
	inline uint32_t get_zone_width(PVZoneID z) const { assert(z < (PVZoneID) _zones_width.size()); return _zones_width[z]; }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	list_zone_images_t& get_zones_images() { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return picviz_min((PVZoneID)(_first_zone + _zones_imgs.size()-1), (PVZoneID)get_zones_manager().get_number_zones()-1); }
	bool is_zone_drawn(PVZoneID z) const { return (z >= get_first_drawn_zone() && z <= get_last_drawn_zone()); }
	uint32_t get_zone_absolute_pos(PVZoneID z) const;

	template <class F>
	inline void set_all_zones_width(F const& f)
	{
		for (PVZoneID zid = 0; zid < (PVZoneID) _zones_width.size(); zid++) {
			set_zone_width(zid, f(get_zone_width(zid)));
		}
	}

private:
	void filter_zone_by_sel_in_task(PVZoneID const z, Picviz::PVSelection const& sel, tbb::task* root);

	PVZoneID get_image_index_of_zone(PVZoneID z) const;

	inline void update_zone_images_width(PVZoneID z)
	{
		assert(is_zone_drawn(z));
		_zones_imgs[z-get_first_drawn_zone()].set_width(get_zone_width(z));
	}
	
	void visit_all_zones_to_render(uint32_t view_width, std::function<void(PVZoneID)> fzone, PVRenderingJob* job = NULL);

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

	void do_translate(PVZoneID pre_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw, PVRenderingJob* job = NULL);

	PVZoneID get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const;

	void left_shift_images(PVZoneID s);
	void right_shift_images(PVZoneID s);

private:
	zones_drawing_t* _zd;
	PVZoneID _first_zone;
	uint32_t _zone_max_width;
	int32_t _visible_view_x;

	std::vector<uint32_t> _zones_width;

	list_zone_images_t _zones_imgs;

	render_group_t _render_grp_sel;
	render_group_t _render_grp_bg;
};

}

#endif
