/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <qtconcurrentrun.h>

#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVTaskFilterSel.h>

#include <tbb/task.h>
#include <tbb/task_group.h>

PVParallelView::PVLinesView::PVLinesView(zones_drawing_t& zd, uint32_t zone_width /* = PVParallelView::ZoneMaxWidth */) :
	_zd(&zd),
	_first_zone(0),
	_zone_max_width(zone_width),
	_visible_view_x(0)
{
	set_nb_drawable_zones(get_zones_manager().get_number_zones());

	_render_grp_sel = zd.new_render_group();
	_render_grp_bg = zd.new_render_group();

	PVLOG_INFO("render_grp_sel: %lu\n", _render_grp_sel);
	PVLOG_INFO("render_grp_bg: %lu\n", _render_grp_bg);

	_zones_width.reserve(get_zones_manager().get_number_zones());
	for (PVZoneID z = 0; z < get_zones_manager().get_number_zones(); z++) {
		_zones_width.emplace_back(PVParallelView::ZoneDefaultWidth);
	}
}

void PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, tbb::task* root_sel, tbb::task_group& grp_bg, const float zoom_y, PVRenderingJob* job)
{
	// First, set new view x (before launching anything in the future !! ;))
	
	PVZoneID pre_first_zone = set_new_view(view_x, view_width);
	if (pre_first_zone == _first_zone) {
		// "Le changement, c'est pas maintenant !"
		PVLOG_INFO("(do_translate) same first zone. Do nothing.\n");
		return;
	}

	do_translate(pre_first_zone, view_width,
	[&](PVZoneID z)
	{
		PVLOG_INFO("(translate) render zone %u\n", z);
		assert(is_zone_drawn(z));
		update_zone_images_width(z);
		grp_bg.run([&,z,job, zoom_y] { this->render_zone_bg(z, zoom_y, job); });
		filter_zone_by_sel_in_task(z, sel, root_sel);
	},
	job);
}

void PVParallelView::PVLinesView::render_zone_all_imgs(PVZoneID z, const Picviz::PVSelection& sel, tbb::task_group& grp_bg, tbb::task* root_sel, const float zoom_y, PVRenderingJob* job)
{
	assert(is_zone_drawn(z));
	update_zone_images_width(z);
	grp_bg.run([&,z,job, zoom_y](){ this->render_zone_bg(z, zoom_y, job); });
	filter_zone_by_sel_in_task(z, sel, root_sel);
}

void PVParallelView::PVLinesView::render_all_zones_all_imgs(int32_t view_x, uint32_t view_width, const Picviz::PVSelection& sel, tbb::task_group& grp_bg, tbb::task* root_sel, const float zoom_y, PVRenderingJob* job_bg)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID z)
	    {
			assert(z >= _first_zone);
			render_zone_all_imgs(z, sel, grp_bg, root_sel, zoom_y, job_bg);
		}
	);
}

void PVParallelView::PVLinesView::render_all_imgs_bg(uint32_t view_width, tbb::task_group& grp_bg, const float zoom_y, PVRenderingJob* job)
{
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID z)
	    {
			assert(z >= _first_zone);
			update_zone_images_width(z);
			grp_bg.run([&,z,job,zoom_y](){ this->render_zone_bg(z, zoom_y, job); });
			//draw_zone_caller_t::call(_zd, *_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci, [=](){ PVLOG_INFO("Zone %d drawn!\n", z); });
		}
	);
}

PVZoneID PVParallelView::PVLinesView::get_image_index_of_zone(PVZoneID z) const
{
	return (is_zone_drawn(z)) ? z-get_first_drawn_zone() : PVZONEID_INVALID;
}

void PVParallelView::PVLinesView::set_nb_drawable_zones(PVZoneID nb_zones)
{
	nb_zones = picviz_min(nb_zones, MaxDrawnZones);
	PVZoneID old_nzones = _zones_imgs.size();
	if (nb_zones == old_nzones || nb_zones <= 0) {
		// Le changement, c'est toujours pas maintenant.
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

void PVParallelView::PVLinesView::left_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _zones_imgs.size());
	std::rotate(_zones_imgs.begin(), _zones_imgs.begin()+s, _zones_imgs.end());
}

void PVParallelView::PVLinesView::filter_zone_by_sel_in_task(PVZoneID const z, Picviz::PVSelection const& sel, tbb::task* root)
{
	root->increment_ref_count();
	tbb::task& child_task = *new (root->allocate_child()) PVTaskFilterSel(get_zones_manager(), z, sel);
	// TODO: add priority
	root->enqueue(child_task);
}

void PVParallelView::PVLinesView::update_sel_tree(uint32_t view_width, const Picviz::PVSelection& sel, tbb::task* root)
{
	visit_all_zones_to_render(view_width,
		[&](PVZoneID z)
		{
			assert(is_zone_drawn(z));
			update_zone_images_width(z);
			this->filter_zone_by_sel_in_task(z, sel, root);
		}
	);
}

void PVParallelView::PVLinesView::render_zone_bg(PVZoneID z, const float zoom_y, PVRenderingJob* job)
{
	assert(is_zone_drawn(z));
	if (_zones_imgs[z-_first_zone].bg->width() != get_zone_width(z)) {
		return;
	}
	_zd->draw_zone(*_zones_imgs[z-_first_zone].bg, 0, z, get_zone_width(z), &PVParallelView::PVZoneTree::browse_tree_bci, zoom_y, []{}, [=](){ job->zone_finished(z); }, _render_grp_bg);
}

void PVParallelView::PVLinesView::render_zone_sel(PVZoneID z, const float zoom_y, PVRenderingJob* job)
{
	assert(is_zone_drawn(z));
	if (_zones_imgs[z-_first_zone].sel->width() != get_zone_width(z)) {
		return;
	}
	_zd->draw_zone(*_zones_imgs[z-_first_zone].sel, 0, z, get_zone_width(z), &PVParallelView::PVZoneTree::browse_tree_bci_sel, zoom_y, []{}, [=](){ job->zone_finished(z); }, _render_grp_sel);
}

void PVParallelView::PVLinesView::visit_all_zones_to_render(uint32_t view_width, std::function<void(PVZoneID)> fzone, PVRenderingJob* job /*= NULL*/)
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
		visit_all_zones_to_render(view_width, fzone_draw, job);
		return;
	}

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

	return ret;
 }

uint32_t PVParallelView::PVLinesView::get_zone_absolute_pos(PVZoneID zone) const
{
	assert(zone < (PVZoneID) _zones_width.size());
	uint32_t pos = 0;
	for (PVZoneID z = 0; z < zone; z++) {
		pos += _zones_width[z] + PVParallelView::AxisWidth;
	}
	return pos;
}

PVZoneID PVParallelView::PVLinesView::get_zone_from_scene_pos(int abs_pos) const
{
	PVZoneID zid = 0;
	ssize_t pos = 0;
	for (; zid < (PVZoneID) (_zones_width.size()-1) ; zid++)
	{
		pos += _zones_width[zid] + PVParallelView::AxisWidth;
		if (pos > abs_pos) {
			break;
		}
	}

	assert(zid < (PVZoneID) _zones_width.size());
	return zid;
}

int PVParallelView::PVLinesView::update_number_of_zones(int view_x, uint32_t view_width)
{
	PVCol old_zones_count = (PVCol) _zones_width.size();
	PVCol new_zones_count = get_zones_manager().get_number_zones();
	set_nb_drawable_zones(new_zones_count);
	_zones_width.resize(new_zones_count, PVParallelView::ZoneDefaultWidth);
	// Update first zone
	set_new_view(view_x, view_width);
	return (int)new_zones_count-(int)old_zones_count;
}
