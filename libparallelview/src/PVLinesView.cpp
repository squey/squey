/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <qtconcurrentrun.h>

#include <pvparallelview/PVLinesView.h>

PVParallelView::PVLinesView::PVLinesView(PVParallelView::PVZonesManager& zm, PVParallelView::PVLinesView::zones_drawing_t::bci_backend_t& bci_backend, PVZoneID nb_zones /*= 30*/, uint32_t zone_width /* = PVParallelView::ZoneMaxWidth */) :
	_zd(new zones_drawing_t(zm, bci_backend, *PVParallelView::PVHSVColor::init_colors(zm.get_number_rows()))),
	_first_zone(0),
	_zone_max_width(zone_width),
	_visible_view_x(0)
{
	set_nb_drawable_zones(nb_zones);
}

QFuture<void> PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job)
{
	PVLOG_INFO("(translate) view_x: %d px\n", view_x);

	// First, set new view x (before launching anything in the future !! ;))
	
	PVZoneID pre_first_zone = set_new_view(view_x, view_width);
	if (pre_first_zone == _first_zone) {
		// "Le changement, c'est pas maintenant !"
		PVLOG_INFO("(do_translate) same first zone. Do nothing.\n");
		return QFuture<void>();
	}

	return QtConcurrent::run([&, pre_first_zone, view_width]
		{
			do_translate(pre_first_zone, view_width,
			[&](PVZoneID z)
			{
				PVLOG_INFO("(translate) render zone %u\n", z);
				assert(z >= _first_zone);
				update_zone_images_width(z);
				PVLOG_INFO("z=%d in _zones_imgs[%d].bg\n", z, z-_first_zone);
				get_zones_manager().filter_zone_by_sel(z, sel);
			},
			&job);
		}
	);
}


void PVParallelView::PVLinesView::render_bg(uint32_t view_width)
{
	render_all_zones(view_width,
	    [&](PVZoneID z)
	    {
			PVLOG_INFO("(render_bg) render zone %u\n", z);
			assert(z >= _first_zone);
			update_zone_images_width(z);
			draw_zone_caller_t::call(_zd, *_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
		}
	);

}

QFuture<void> PVParallelView::PVLinesView::render_sel(uint32_t view_width, PVRenderingJob& job)
{
	return QtConcurrent::run<>([&, view_width]{
			render_all_zones(view_width,
				[&](PVZoneID z)
				{
					PVLOG_INFO("(render_all_imgs) render zone %u\n", z);
					assert(is_zone_drawn(z));
					update_zone_images_width(z);
					draw_zone_sel_caller_t::call(_zd, *_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
				},
				&job
			);
		}
	);
}

QFuture<void> PVParallelView::PVLinesView::render_all_imgs(uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job)
{
	return QtConcurrent::run<>([&, view_width]{
			render_all_zones(view_width,
				[&](PVZoneID z)
				{
					PVLOG_INFO("(render_all_imgs) render zone %u\n", z);
					assert(z >= _first_zone);
					update_zone_images_width(z);
					get_zones_manager().filter_zone_by_sel(z, sel);
					//draw_zone_caller_t::call(_zd, *_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
					//draw_zone_sel_caller_t::call(_zd, *_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
				},
				&job
			);
		}
	);
}


/*bool PVParallelView::PVLinesView::set_zone_width_and_render(PVZoneID zid, uint32_t width)
{
	if (!set_zone_width(zid, width)) {
		// width hasn't changed !
		return false;
	}

	if (is_zone_drawn(zid)) {
		render_zone_all_imgs(zid);
		return true;
	}

	return false;
}*/

QFuture<void> PVParallelView::PVLinesView::render_zone_all_imgs(PVZoneID z, const Picviz::PVSelection& sel, PVRenderingJob& job)
{
	PVLOG_INFO("(lines view) render zone %d\n", z);
	if (!is_zone_drawn(z)) {
		// "You've got no future" !
		return QFuture<void>();
	}

	PVZoneID img_id = z-_first_zone;
	return QtConcurrent::run<>([&, z, img_id]
		{
			update_zone_images_width(z);
			get_zones_manager().filter_zone_by_sel(z, sel);
			//draw_zone_caller_t::call(_zd, *_zones_imgs[img_id].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
			//draw_zone_sel_caller_t::call(_zd, *_zones_imgs[img_id].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
			//job.zone_finished(z);
		}
	);
}

PVZoneID PVParallelView::PVLinesView::get_image_index_of_zone(PVZoneID z) const
{
	return (is_zone_drawn(z)) ? z-get_first_drawn_zone() : PVZONEID_INVALID;
}

void PVParallelView::PVLinesView::set_nb_drawable_zones(PVZoneID nb_zones)
{
	PVZoneID old_nzones = _zones_imgs.size();
	if (nb_zones == old_nzones || nb_zones <= 0) {
		return;
	}

	if (nb_zones > old_nzones) {
		const PVZoneID nnew = nb_zones-old_nzones;
		for (PVZoneID z = 0; z < nnew; z++) {
			_zones_imgs.push_back(ZoneImages(_zd, _zone_max_width));
		}
	}
	else {
		_zones_imgs.resize(nb_zones);
	}
}

void PVParallelView::PVLinesView::set_zone_max_width(uint32_t w)
{
	list_zone_images_t::iterator it;
	for (it = _zones_imgs.begin(); it != _zones_imgs.end(); it++) {
		it->create_image(_zd, w);
	}
}

#if 0
PVZoneID PVParallelView::PVLinesView::get_new_first_zone_with_translation(int32_t vec_x)
{
	// TODO: that may be optimizable !
	int64_t first_zone_x = get_zone_absolute_pos(_first_zone);
	first_zone_x += vec_x;
	if (first_zone_x <= 0) {
		return 0;
	}
	PVZoneID ret = get_zone_from_scene_pos(first_zone_x);
	if (ret == PVZONEID_INVALID) {
		PVZoneID nzones_img = _zones_imgs.size();
		const PVZoneID nzones = get_zones_manager().get_number_zones();
		if (nzones_img >= nzones) {
			ret = 0;
		}
		else {
			ret = nzones-nzones_img;
		}
	}
	return ret;
}
#endif

void PVParallelView::PVLinesView::left_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _zones_imgs.size());
	std::rotate(_zones_imgs.begin(), _zones_imgs.begin()+s, _zones_imgs.end());
}

QFuture<void> PVParallelView::PVLinesView::render_all(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, PVRenderingJob& job)
{
	set_new_view(view_x, view_width);
	return render_all_imgs(view_width, sel, job);
}

QFuture<void> PVParallelView::PVLinesView::update_sel_from_zone(uint32_t view_width, PVZoneID zid_sel, const Picviz::PVSelection& sel, PVRenderingJob& job)
{
	// Flag the selection as invalid (_sel_elts from all PVZoneTree are flagged as invalid)
	get_zones_manager().invalidate_selection();

	return QtConcurrent::run<>([&, view_width, zid_sel] {
		render_all_zones(view_width,
			[&, view_width, zid_sel](PVZoneID z)
			{
				PVLOG_INFO("(render_sel) render zone %u\n", z);
				assert(is_zone_drawn(z));
				update_zone_images_width(z);
				if (zid_sel != z) {
					get_zones_manager().filter_zone_by_sel(z, sel);
				}
			},
			&job
		);
		}
	);
}

void PVParallelView::PVLinesView::draw_zone(PVZoneID z)
{
	PVLOG_INFO("draw_zone_caller_t %d\n", z);
	draw_zone_caller_t::call(_zd, *_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
	draw_zone_sel(z);
}

void PVParallelView::PVLinesView::draw_zone_sel(PVZoneID z)
{
	draw_zone_sel_caller_t::call(_zd, *_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
}

void PVParallelView::PVLinesView::render_all_zones(uint32_t view_width, std::function<void(PVZoneID)> fzone, PVRenderingJob* job /*= NULL*/)
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
			right_invisible_zone++;
			zones_to_draw--;
			one_done = true;
		}
		if (!one_done) {
			break;
		}
	}
}

void PVParallelView::PVLinesView::do_translate(PVZoneID pre_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw, PVRenderingJob* job /*= NULL*/)
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
		}
	}
}

void PVParallelView::PVLinesView::right_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _zones_imgs.size());
	if (s > 0) {
		std::rotate(_zones_imgs.begin(), _zones_imgs.end()-s, _zones_imgs.end());
	}
}

PVZoneID PVParallelView::PVLinesView::get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const
{
	if (view_x < 0) {
		uint32_t unused_width = (uint32_t) (-view_x);
		if (unused_width >= view_width) {
			unused_width = view_width;
		}
		view_width -= unused_width;
		view_x = 0;
	}

	const PVZoneID nzones_total = get_zones_manager().get_number_zones();
	const PVZoneID zfirst_visible = get_zone_from_scene_pos(view_x);
	PVZoneID zones_drawable = _zones_imgs.size();

	uint32_t cur_width = 0;
	PVZoneID cur_z = zfirst_visible;
	while (cur_width < view_width && cur_z < nzones_total) {
		const uint32_t offset = get_zone_width(cur_z) + PVParallelView::AxisWidth;
		cur_width += offset;
		cur_z++;
		zones_drawable--;
	}

	if (cur_z >= nzones_total) {
		// All remaining zones are going to the left
		return std::max(0, zfirst_visible-zones_drawable);
	}

	// 'zones_drawable' can now be considered as "secure" zones.
	PVZoneID ret;
	if ((zones_drawable & 1) == 0) {
		ret = zfirst_visible - (zones_drawable/2);
	}
	else {
		ret = zfirst_visible - (zones_drawable/2) - 1;
	}
	ret = PVCore::clamp(ret, 0, (PVZoneID) std::max(0, (nzones_total-(PVZoneID)_zones_imgs.size())));
	PVLOG_INFO("From viewport %d/%u: first zone %d\n", view_x, view_width, ret);

	return ret;
 }
