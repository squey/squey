#include <pvparallelview/PVLinesView.h>
#include <qtconcurrentrun.h>

PVParallelView::PVLinesView::PVLinesView(PVZonesDrawing& zones_drawing, PVZoneID nb_zones, uint32_t zone_width /* = PVParallelView::ZoneMaxWidth */) :
	_zd(zones_drawing),
	_first_zone(0),
	_zone_max_width(zone_width),
	_visible_view_x(0)
{
	set_nb_drawable_zones(nb_zones);
}

void PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width)
{
	PVLOG_INFO("(translate) view_x: %d px\n", view_x);

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
			assert(z >= _first_zone);
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
		}
	);
}

QFuture<void> PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, PVRenderingJob& job)
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
				_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
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
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
		}
	);

}

void PVParallelView::PVLinesView::render_sel(uint32_t view_width)
{
	render_all_zones(view_width,
		[&](PVZoneID z)
		{
			PVLOG_INFO("(render_sel) render zone %u\n", z);
			assert(is_zone_drawn(z));
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
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
					assert(z >= _first_zone);
					_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
				},
				&job
			);
		}
	);
}

void PVParallelView::PVLinesView::render_all_imgs(uint32_t view_width)
{
	render_all_zones(view_width,
		[&](PVZoneID z)
		{
			PVLOG_INFO("(render_all_imgs) render zone %u\n", z);
			assert(is_zone_drawn(z));
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
		}
	);
}

QFuture<void> PVParallelView::PVLinesView::render_all_imgs(uint32_t view_width, PVRenderingJob& job)
{
	return QtConcurrent::run<>([&, view_width]{
			render_all_zones(view_width,
				[&](PVZoneID z)
				{
					PVLOG_INFO("(render_all_imgs) render zone %u\n", z);
					assert(z >= _first_zone);
					_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
					_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].sel, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci_sel);
				},
				&job
			);
		}
	);
}


bool PVParallelView::PVLinesView::set_zone_width_and_render(PVZoneID zid, uint32_t width)
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
}

void PVParallelView::PVLinesView::render_zone_all_imgs(PVZoneID z)
{
	PVLOG_INFO("(lines view) render zone %d\n", z);
	if (is_zone_drawn(z)) {
		_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[z-_first_zone].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
	}
}

QFuture<void> PVParallelView::PVLinesView::render_zone_all_imgs(PVZoneID z, PVRenderingJob& job)
{
	PVLOG_INFO("(lines view) render zone %d\n", z);
	if (!is_zone_drawn(z)) {
		// "You've got no future" !
		return QFuture<void>();
	}

	PVZoneID img_id = z-_first_zone;
	return QtConcurrent::run<>([&, z, img_id]
		{
			_zd.draw_zone<PVParallelView::PVZoneTree>(*_zones_imgs[img_id].bg, 0, z, &PVParallelView::PVZoneTree::browse_tree_bci);
			job.zone_finished(z);
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

void PVParallelView::PVLinesView::render_all(int32_t view_x, uint32_t view_width)
{
	set_new_view(view_x, view_width);
	render_all_imgs(view_width);
}

QFuture<void> PVParallelView::PVLinesView::render_all(int32_t view_x, uint32_t view_width, PVRenderingJob& job)
{
	set_new_view(view_x, view_width);
	return render_all_imgs(view_width, job);
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
