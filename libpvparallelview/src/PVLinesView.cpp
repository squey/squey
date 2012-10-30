/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZoneTree.h>

#include <QObject>
#include <QMetaMethod>

PVParallelView::PVLinesView::PVLinesView(PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg, QObject* img_update_receiver, uint32_t zone_width):
	_first_zone(0),
	_zone_max_width(zone_width),
	_visible_view_x(0),
	_processor_sel(zp_sel),
	_processor_bg(zp_bg),
	_zm(zm),
	_backend(backend),
	_img_update_receiver(img_update_receiver)
{
	set_nb_drawable_zones(get_zones_manager().get_number_zones());


	_zones_width.resize(get_zones_manager().get_number_zones(), PVParallelView::ZoneDefaultWidth);
}

void PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	// First, set new view x (before launching anything in the future !! ;))
	
	PVZoneID pre_first_zone = set_new_view(view_x, view_width);
	if (pre_first_zone == _first_zone) {
		// "Le changement, c'est pas maintenant !"
		return;
	}

	do_translate(pre_first_zone, view_width,
	[&](PVZoneID z)
	{
		assert(is_zone_drawn(z));
		render_zone_all_imgs(z, zoom_y);
	});
}

void PVParallelView::PVLinesView::render_zone_all_imgs(PVZoneID z, const float zoom_y)
{
	assert(is_zone_drawn(z));
	render_zone_bg(z, zoom_y);
	render_zone_sel(z, zoom_y);
}

void PVParallelView::PVLinesView::render_all_zones_all_imgs(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID z)
	    {
			assert(is_zone_drawn(z));
			render_zone_all_imgs(z, zoom_y);
		}
	);
}

void PVParallelView::PVLinesView::render_all_imgs_bg(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID z)
	    {
			assert(is_zone_drawn(z));
			render_zone_bg(z, zoom_y);
		}
	);
}

void PVParallelView::PVLinesView::render_all_imgs_sel(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID z)
	    {
			assert(is_zone_drawn(z));
			render_zone_sel(z, zoom_y);
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
			_zones_imgs.emplace_back(this->backend(), _zone_max_width);
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
		it->create_image(backend(), w);
	}
}

void PVParallelView::PVLinesView::left_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _zones_imgs.size());
	std::rotate(_zones_imgs.begin(), _zones_imgs.begin()+s, _zones_imgs.end());
}

void PVParallelView::PVLinesView::render_zone_bg(PVZoneID z, const float zoom_y)
{
	assert(is_zone_drawn(z));

	ZoneImages& zi = get_zone_images(z);
	zi.cancel_last_bg();
	const uint32_t width = get_zone_width(z);
	zi.bg->set_width(width);

	PVZoneRendering<PARALLELVIEW_ZT_BBITS>* zr = new (PVRenderingPipeline::allocate_zr<PARALLELVIEW_ZT_BBITS>()) PVZoneRendering<PARALLELVIEW_ZT_BBITS>(z,
		[&,width,zoom_y](PVZoneID z, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(z).browse_tree_bci(colors, codes);
		},
		*zi.bg,
		0,
		width,
		zoom_y,
		false // not reversed
		);

	connect_zr(zr, "zr_bg_finished");
	zi.last_zr_bg = zr;

	_processor_bg.add_job(*zr);
}

void PVParallelView::PVLinesView::render_zone_sel(PVZoneID z, const float zoom_y)
{
	assert(is_zone_drawn(z));

	ZoneImages& zi = get_zone_images(z);
	zi.cancel_last_sel();
	const uint32_t width = get_zone_width(z);
	zi.sel->set_width(width);

	PVZoneRendering<PARALLELVIEW_ZT_BBITS>* zr = new (PVRenderingPipeline::allocate_zr<PARALLELVIEW_ZT_BBITS>()) PVZoneRendering<PARALLELVIEW_ZT_BBITS>(z,
		[&,width,zoom_y](PVZoneID z, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(z).browse_tree_bci_sel(colors, codes);
		},
		*zi.sel,
		0,
		width,
		zoom_y,
		false // not reversed
		);

	connect_zr(zr, "zr_sel_finished");
	zi.last_zr_sel = zr;

	_processor_sel.add_job(*zr);
}

void PVParallelView::PVLinesView::connect_zr(PVZoneRenderingBase* zr, const char* slot)
{
	if (_img_update_receiver) {
		zr->set_render_finished_slot(_img_update_receiver, slot);
	}
}

void PVParallelView::PVLinesView::visit_all_zones_to_render(uint32_t view_width, std::function<void(PVZoneID)> const& fzone)
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
			fzone(left_invisible_zone);
			zones_to_draw--;
			if (zones_to_draw == 0) {
				break;
			}
			one_done = true;
		}
		if (right_invisible_zone < nzones_total) {
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

void PVParallelView::PVLinesView::do_translate(PVZoneID pre_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw)
{
	const PVZoneID nzones_img = _zones_imgs.size();
	const PVZoneID diff = std::abs(_first_zone - pre_first_zone);
	if (diff >= nzones_img) {
		visit_all_zones_to_render(view_width, fzone_draw);
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
		/*if (job) {
			for (PVZoneID z = _first_zone; z < first_z_to_render; z++) {
				job->zone_finished(z);
			}
		}*/
		if (_img_update_receiver) {
			for (PVZoneID z = _first_zone; z < first_z_to_render; z++) {
				call_refresh_slots(z);
			}
		}

		for (PVZoneID z = first_z_to_render; z < last_z; z++) {
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
		/*if (job) {
			for (PVZoneID z = last_z; z <= get_last_drawn_zone(); z++) {
				job->zone_finished(z);
			}
		}*/
		if (_img_update_receiver) {
			for (PVZoneID z = last_z; z <= get_last_drawn_zone(); z++) {
				call_refresh_slots(z);
			}
		}

		for (PVZoneID z = last_z-1; z >= first_z_to_render; z--) {
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
	int zones_drawable = _zones_imgs.size();

	uint32_t cur_width = 0;
	PVZoneID cur_z = zfirst_visible;
	while (cur_width < view_width && cur_z < nzones_total && zones_drawable > 0) {
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

PVZoneID PVParallelView::PVLinesView::get_number_zones() const
{
	return get_zones_manager().get_number_zones();
}

void PVParallelView::PVLinesView::cancel_and_wait_all_rendering()
{
	for (ZoneImages& zi: _zones_imgs) {
		zi.cancel_all_and_wait();
	}
}

bool PVParallelView::PVLinesView::set_zone_width(PVZoneID z, uint32_t width)
{
	assert(z < (PVZoneID) _zones_width.size());
	// Returns true if width was actual changed
	uint32_t old_width = get_zone_width(z);
	uint32_t new_width = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
	bool diff = new_width != old_width;
	if (diff) {
		_zones_width[z] = new_width;
	}
	return diff;
}

void PVParallelView::PVLinesView::call_refresh_slots(int zid)
{
	// Call both zr_sel_finished and zr_bg_finished slots on _img_update_receiver
	if (!_img_update_receiver) {
		return;
	}

	QMetaObject::invokeMethod(_img_update_receiver, "zr_sel_finished", Qt::QueuedConnection,
			Q_ARG(void*, NULL),
			Q_ARG(int, zid));
	QMetaObject::invokeMethod(_img_update_receiver, "zr_bg_finished",  Qt::QueuedConnection,
			Q_ARG(void*, NULL),
			Q_ARG(int, zid));
}

// ZoneImages implementation
//

void PVParallelView::PVLinesView::ZoneImages::create_image(PVBCIDrawingBackend& backend, uint32_t width)
{
	sel = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
	bg = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
}

void PVParallelView::PVLinesView::ZoneImages::set_width(uint32_t width)
{
	sel->set_width(width);
	bg->set_width(width);
}

void PVParallelView::PVLinesView::ZoneImages::cancel_last_sel()
{
	if (last_zr_sel) {
		last_zr_sel->cancel();
		//last_zr_sel->wait_end();
	}
}

void PVParallelView::PVLinesView::ZoneImages::cancel_last_bg()
{
	if (last_zr_bg) {
		last_zr_bg->cancel();
		//last_zr_bg->wait_end();
	}
}

void PVParallelView::PVLinesView::ZoneImages::cancel_all_and_wait()
{
	cancel_last_bg();
	cancel_last_sel();
}
