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
}

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
	typedef PVCore::PVSharedPtr<zones_drawing_t> zones_drawing_sp;
	typedef typename zones_drawing_t::backend_image_p_t backend_image_p_t;

private:
	struct ZoneImages
	{
		ZoneImages() { }

		ZoneImages(PVParallelView::PVLinesView::zones_drawing_sp zd, uint32_t zone_width)
		{
			create_image(zd, zone_width);
		}

		void set_width(uint32_t zone_width)
		{
			sel->set_width(zone_width);
			bg->set_width(zone_width);
		}

		void create_image(PVParallelView::PVLinesView::zones_drawing_sp zd, uint32_t zone_width)
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
	PVLinesView(PVParallelView::PVZonesManager& zm, PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend, PVZoneID nb_drawable_zones = 30, uint32_t zone_width = PVParallelView::ZoneMaxWidth);

public:
	void set_nb_drawable_zones(PVZoneID nb_zones);
	void set_zone_max_width(uint32_t w);

public:
	QFuture<void> translate(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job);

	QFuture<void> render_all(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job);

	QFuture<void> render_all_imgs(uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job);

	void render_bg(uint32_t view_width);

	QFuture<void> render_sel(uint32_t view_width, PVRenderingJob& job);

	void update_sel_tree(uint32_t view_width, const Picviz::PVSelection& sel, tbb::task* root);

	QFuture<void> render_zone_all_imgs(PVZoneID z, const Picviz::PVSelection& sel, PVRenderingJob& job);

	inline PVZoneID get_zone_from_scene_pos(int32_t x) const { return get_zones_manager().get_zone_id(x); }

	inline bool set_zone_width(PVZoneID z, uint32_t width)
	{
		// Returns true if width was actual changed
		uint32_t old_width = get_zone_width(z);
		get_zones_manager().set_zone_width(z, width);
		return get_zone_width(z) != old_width;
	}
	//bool set_zone_width_and_render(PVZoneID z, uint32_t width);

	inline zones_drawing_sp& get_zones_drawing() { return _zd; }
	inline const PVZonesManager& get_zones_manager() const { return _zd->get_zones_manager(); }
	inline PVZonesManager& get_zones_manager() { return _zd->get_zones_manager(); }
	inline uint32_t get_zone_width(PVZoneID z) const { return _zd->get_zone_width(z); }

	const list_zone_images_t& get_zones_images() const { return _zones_imgs; }
	list_zone_images_t& get_zones_images() { return _zones_imgs; }
	inline PVZoneID get_first_drawn_zone() const { return _first_zone; }
	inline PVZoneID get_last_drawn_zone() const { return picviz_min((PVZoneID)(_first_zone + _zones_imgs.size()-1), (PVZoneID)get_zones_manager().get_number_zones()-1); }
	bool is_zone_drawn(PVZoneID z) const { return (z >= get_first_drawn_zone() && z <= get_last_drawn_zone()); }
	inline uint32_t get_zone_absolute_pos(PVZoneID z) const { return get_zones_manager().get_zone_absolute_pos(z); }

	template <class F>
	inline void set_all_zones_width(F const& f) { get_zones_manager().set_zones_width(f); }

	void draw_zone(PVZoneID z);
	void draw_zone_sel(PVZoneID z);

	/*template <class F>
	bool set_all_zones_width_and_render(int32_t visible_view_x, uint32_t width, F const& f)
	{
		//width = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
		get_zones_manager().set_zones_width(f);
		render_all(visible_view_x, width);
		return true;
	}*/

private:
	PVZoneID get_image_index_of_zone(PVZoneID z) const;

	inline void update_zone_images_width(PVZoneID z)
	{
		assert(is_zone_drawn(z));
		_zones_imgs[z-get_first_drawn_zone()].set_width(get_zone_width(z));
	}
	
	void render_all_zones(uint32_t view_width, std::function<void(PVZoneID)> fzone, PVRenderingJob* job = NULL);

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
	PVParallelView::PVLinesView::zones_drawing_sp _zd;
	PVZoneID _first_zone;
	uint32_t _zone_max_width;
	int32_t _visible_view_x;

	list_zone_images_t _zones_imgs;
	typedef PVHive::PVHiveFuncCaller<FUNC(PVLinesView::zones_drawing_t::draw_zone<decltype(&PVParallelView::PVZoneTree::browse_tree_bci_sel)>)> draw_zone_sel_caller_t;
	typedef PVHive::PVHiveFuncCaller<FUNC(PVLinesView::zones_drawing_t::draw_zone<decltype(&PVParallelView::PVZoneTree::browse_tree_bci)>)> draw_zone_caller_t;
};

}

#endif
