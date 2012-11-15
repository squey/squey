/**
 * \file PVLinesView.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <cmath>

#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVLinesView.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZoneTree.h>

#include <QObject>
#include <QMetaMethod>

/******************************************************************************
 *
 * PVParallelView::PVLinesView::PVLinesView
 *
 *****************************************************************************/
PVParallelView::PVLinesView::PVLinesView(PVBCIDrawingBackend& backend, PVZonesManager const& zm, PVZonesProcessor& zp_sel, PVZonesProcessor& zp_bg, QObject* img_update_receiver, uint32_t zone_width):
	_backend(backend),
	_first_zone(0),
	_global_zoom_level(0),
	_img_update_receiver(img_update_receiver),
	_processor_sel(zp_sel),
	_processor_bg(zp_bg),
	_visible_view_x(0),
	_zm(zm),
	_zone_max_width(zone_width)
{
	set_nb_drawable_zones(get_number_of_managed_zones());

	//We initialize all zones width
	_zones_width.resize(get_number_of_managed_zones(), PVParallelView::ZoneDefaultWidth);
	//We initialize all zones ZoneWidthWithZoomLevel
	_list_of_zone_width_with_zoom_level.resize(get_number_of_managed_zones());
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::call_refresh_slots
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::call_refresh_slots(PVZoneID zone_id)
{
	// Call both zr_sel_finished and zr_bg_finished slots on _img_update_receiver
	if (!_img_update_receiver) {
		return;
	}

	QMetaObject::invokeMethod(_img_update_receiver, "zr_sel_finished", Qt::QueuedConnection,
			Q_ARG(PVParallelView::PVZoneRenderingBase_p, PVZoneRenderingBase_p()),
			Q_ARG(int, (int) zone_id));
	QMetaObject::invokeMethod(_img_update_receiver, "zr_bg_finished",  Qt::QueuedConnection,
			Q_ARG(PVParallelView::PVZoneRenderingBase_p, PVZoneRenderingBase_p()),
			Q_ARG(int, (int) zone_id));
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::cancel_and_wait_all_rendering
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::cancel_and_wait_all_rendering()
{
	for (SingleZoneImages& single_zone_images: _list_of_single_zone_images) {
		single_zone_images.cancel_all_and_wait();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::connect_zr
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::connect_zr(PVZoneRenderingBase* zr, const char* slot)
{
	if (_img_update_receiver) {
		zr->set_render_finished_slot(_img_update_receiver, slot);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::decrease_base_zoom_level_of_zone
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::decrease_base_zoom_level_of_zone(PVZoneID zone_id)
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	
	_list_of_zone_width_with_zoom_level[zone_id].decrease_zoom_level();
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::decrease_global_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::decrease_global_zoom_level()
{
	if (_global_zoom_level > ZoneWidthWithZoomLevel::min_zoom_level) {
		--_global_zoom_level;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::do_translate
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::do_translate(PVZoneID previous_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw)
{
	const PVZoneID number_of_visible_zones = get_number_of_visible_zones();
	const PVZoneID unsigned_translation_offset = std::abs(_first_zone - previous_first_zone);
	
	if (unsigned_translation_offset >= number_of_visible_zones) {
		visit_all_zones_to_render(view_width, fzone_draw);
		return;
	}

	// We test whether translation happened on the left or on the right.
	if (_first_zone > previous_first_zone) {
		// The scene was translated to the left
		const PVZoneID n = unsigned_translation_offset;
		left_rotate_single_zone_images(n);

		const PVZoneID nimgs = get_number_of_visible_zones();
		PVZoneID first_z_to_render = _first_zone + nimgs - n;
		const PVZoneID last_z = picviz_min(nimgs+_first_zone, get_number_of_managed_zones());

		// If a rendering job is provided, tell him that we virtually have rendered from _first_zone to first_z_to_render images
		/*if (job) {
			for (PVZoneID z = _first_zone; z < first_z_to_render; z++) {
				job->zone_finished(z);
			}
		}*/
		if (_img_update_receiver) {
			for (PVZoneID zone_id = _first_zone; zone_id < first_z_to_render; zone_id++) {
				call_refresh_slots(zone_id);
			}
		}

		for (PVZoneID zone_id = first_z_to_render; zone_id < last_z; zone_id++) {
			fzone_draw(zone_id);
		}
	}
	else {
		// The scene was translated to the right
		const PVZoneID n = unsigned_translation_offset;
		right_rotate_single_zone_images(unsigned_translation_offset);
		PVZoneID first_z_to_render = _first_zone;
		const PVZoneID last_z = picviz_min(_first_zone + n, get_number_of_managed_zones());

		// If a rendering job is provided, tell him that we virtually have rendered from last_z to get_last_visible_zone_index()
		/*if (job) {
			for (PVZoneID z = last_z; z <= get_last_visible_zone_index(); z++) {
				job->zone_finished(z);
			}
		}*/
		if (_img_update_receiver) {
			for (PVZoneID zone_id = last_z; zone_id <= get_last_visible_zone_index(); zone_id++) {
				call_refresh_slots(zone_id);
			}
		}

		for (PVZoneID zone_id = last_z-1; zone_id >= first_z_to_render; zone_id--) {
			fzone_draw(zone_id);
		}
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::update_and_get_first_zone_from_viewport
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::update_and_get_first_zone_from_viewport(int32_t view_x, uint32_t view_width) const
{
	// We test whether the viewport has some unused area on the left, before any zones.
	if (view_x < 0) {
		// There is some empty area on the left.
		uint32_t unused_width = (uint32_t) (-view_x);
		// THIS SHOULD NEVER HAPPEN : We test if the viewport is completely empty of zones (everything pushed too far on the right...)
		if (unused_width >= view_width) {
			// If so, we limit the offset so that it stops just at the rightmost limit
			unused_width = view_width;
		}
		
		// ??? 
		view_width -= unused_width;
		view_x = 0;
	}

	// We init some counters
	const PVZoneID total_number_of_zones = get_number_of_managed_zones();
	const PVZoneID zone_index_of_first_visible_zone = get_zone_from_scene_pos(view_x);
	int counter_of_visible_zones_to_draw = get_number_of_visible_zones();

	// We start 
	uint32_t current_width = 0;
	PVZoneID current_zone_index = zone_index_of_first_visible_zone;
	while (current_width < view_width && current_zone_index < total_number_of_zones && counter_of_visible_zones_to_draw > 0) {
		const uint32_t offset = get_zone_width(current_zone_index) + PVParallelView::AxisWidth;
		current_width += offset;
		current_zone_index++;
		counter_of_visible_zones_to_draw--;
	}

	if (current_zone_index >= total_number_of_zones) {
		// All remaining zones are going to the left
		return std::max(0, zone_index_of_first_visible_zone-counter_of_visible_zones_to_draw);
	}

	// 'counter_of_visible_zones_to_draw' can now be considered as "secure" zones.
	PVZoneID ret;
	if ((counter_of_visible_zones_to_draw & 1) == 0) {
		ret = zone_index_of_first_visible_zone - (counter_of_visible_zones_to_draw/2);
	}
	else {
		ret = zone_index_of_first_visible_zone - (counter_of_visible_zones_to_draw/2) - 1;
	}
	
	ret = PVCore::clamp(ret, 0, (PVZoneID) std::max(0, (total_number_of_zones-(PVZoneID)get_number_of_visible_zones())));

	return ret;
 }

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_image_index_of_zone
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_image_index_of_zone(PVZoneID zone_id) const
{
	return (is_zone_drawn(zone_id)) ? zone_id-get_first_visible_zone_index() : PVZONEID_INVALID;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_left_border_position_of_zone_in_scene
 *
 *****************************************************************************/
uint32_t PVParallelView::PVLinesView::get_left_border_position_of_zone_in_scene(PVZoneID zone_id) const
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	
	// The first zone start in scene at absciss 0
	uint32_t pos = 0;
	
	// We do stop after the right axis of the previous zone
	for (PVZoneID zid = 0; zid < zone_id; zid++) {
		pos += get_zone_width(zid) + PVParallelView::AxisWidth;
	}
	
	return pos;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_number_of_managed_zones
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_number_of_managed_zones() const
{
	return get_zones_manager().get_number_of_managed_zones();
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_right_border_position_of_zone_in_scene
 *
 *****************************************************************************/
uint32_t PVParallelView::PVLinesView::get_right_border_position_of_zone_in_scene(PVZoneID zone_id) const
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	// FIXME : where is the first Axis ? 0, negative or more on the right ???
	// The first zone start in scene at absciss 0
	uint32_t pos = 0;
	
	// We do stop after the right axis of the previous zone
	for (PVZoneID zid = 0; zid < zone_id; zid++) {
		pos += get_zone_width(zid) + PVParallelView::AxisWidth;
	}
	
	pos += get_zone_width(zone_id);
	
	return pos;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_zone_from_scene_pos
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_zone_from_scene_pos(int abs_pos) const
{
	// This computes a zone OFFSET !!! ???
	PVZoneID zone_id = 0;
	ssize_t pos = 0;
	for (; zone_id < (PVZoneID) (_zones_width.size()-1) ; zone_id++)
	{
		pos += get_zone_width(zone_id) + PVParallelView::AxisWidth;
		if (pos > abs_pos) {
			break;
		}
	}

	assert(zone_id < (PVZoneID) _zones_width.size());
	return zone_id;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_zone_width
 *
 *****************************************************************************/
uint32_t PVParallelView::PVLinesView::get_zone_width(PVZoneID zone_id) const
 {
	 assert(zone_id < (PVZoneID) _zones_width.size());
	 
	 uint32_t width = _list_of_zone_width_with_zoom_level[zone_id].get_width(_global_zoom_level);
	 
	 return width;	
}
 
/******************************************************************************
 *
 * PVParallelView::PVLinesView::increase_base_zoom_level_of_zone
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::increase_base_zoom_level_of_zone(PVZoneID zone_id)
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	
	_list_of_zone_width_with_zoom_level[zone_id].increase_zoom_level();
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::increase_global_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::increase_global_zoom_level()
{
	if (_global_zoom_level <= ZoneWidthWithZoomLevel::max_zoom_level) {
		++_global_zoom_level;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::left_rotate_single_zone_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::left_rotate_single_zone_images(PVZoneID s)
{
	assert(s < (PVZoneID) get_number_of_visible_zones());
	std::rotate(_list_of_single_zone_images.begin(), _list_of_single_zone_images.begin()+s, _list_of_single_zone_images.end());
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_all_zones_bg_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_all_zones_bg_image(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID zone_id)
	    {
			assert(is_zone_drawn(zone_id));
			render_single_zone_bg_image(zone_id, zoom_y);
		}
	);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_all_zones_sel_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_all_zones_sel_image(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID zone_id)
	    {
			assert(is_zone_drawn(zone_id));
			render_single_zone_sel_image(zone_id, zoom_y);
		}
	);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_single_zone_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_single_zone_images(PVZoneID zone_id, const float zoom_y)
{
	assert(is_zone_drawn(zone_id));
	render_single_zone_bg_image(zone_id, zoom_y);
	render_single_zone_sel_image(zone_id, zoom_y);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_all_zones_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_all_zones_images(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	set_new_view(view_x, view_width);
	visit_all_zones_to_render(view_width,
	    [&](PVZoneID zone_id)
	    {
			assert(is_zone_drawn(zone_id));
			render_single_zone_images(zone_id, zoom_y);
		}
	);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_single_zone_bg_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_single_zone_bg_image(PVZoneID zone_id, const float zoom_y)
{
	assert(is_zone_drawn(zone_id));
	assert(QThread::currentThread() == _img_update_receiver->thread());

	SingleZoneImages& single_zone_images = get_single_zone_images(zone_id);
	single_zone_images.cancel_last_bg();
	const uint32_t width = get_zone_width(zone_id);
	single_zone_images.bg->set_width(width);

	PVZoneRendering_p<PARALLELVIEW_ZT_BBITS> zr(new PVZoneRendering<PARALLELVIEW_ZT_BBITS>(zone_id,
		[&](PVZoneID zone_id, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(zone_id).browse_tree_bci(colors, codes);
		},
		*single_zone_images.bg,
		0,
		width,
		zoom_y,
		false // not reversed
		));

	connect_zr(zr.get(), "zr_bg_finished");
	single_zone_images.last_zr_bg = zr;

	bool ret = _processor_bg.add_job(zr);
	assert(ret);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_single_zone_sel_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_single_zone_sel_image(PVZoneID zone_id, const float zoom_y)
{
	assert(is_zone_drawn(zone_id));
	assert(QThread::currentThread() == _img_update_receiver->thread());

	SingleZoneImages& single_zone_images = get_single_zone_images(zone_id);
	single_zone_images.cancel_last_sel();
	const uint32_t width = get_zone_width(zone_id);
	single_zone_images.sel->set_width(width);

	PVZoneRendering_p<PARALLELVIEW_ZT_BBITS> zr(new PVZoneRendering<PARALLELVIEW_ZT_BBITS>(zone_id,
		[&](PVZoneID zone_id, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(zone_id).browse_tree_bci_sel(colors, codes);
		},
		*single_zone_images.sel,
		0,
		width,
		zoom_y,
		false // not reversed
		));

	connect_zr(zr.get(), "zr_sel_finished");
	single_zone_images.last_zr_sel = zr;

	bool ret = _processor_sel.add_job(zr);
	assert(ret);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::right_rotate_single_zone_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::right_rotate_single_zone_images(PVZoneID s)
{
	assert(s < (PVZoneID) get_number_of_visible_zones());
	if (s > 0) {
		std::rotate(_list_of_single_zone_images.begin(), _list_of_single_zone_images.end()-s, _list_of_single_zone_images.end());
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::set_nb_drawable_zones
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::set_nb_drawable_zones(PVZoneID nb_zones)
{
	nb_zones = picviz_min(nb_zones, MaxDrawnZones);
	PVZoneID old_nzones = get_number_of_visible_zones();
	if (nb_zones == old_nzones || nb_zones <= 0) {
		// Le changement, c'est toujours pas maintenant.
		return;
	}

	if (nb_zones > old_nzones) {
		const PVZoneID number_of_new_zones = nb_zones-old_nzones;
		for (PVZoneID z = 0; z < number_of_new_zones; z++) {
			_list_of_single_zone_images.emplace_back(this->backend(), _zone_max_width);
		}
	}
	else {
		_list_of_single_zone_images.resize(nb_zones);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::set_zone_max_width
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::set_zone_max_width(uint32_t w)
{
	list_zone_images_t::iterator it;
	for (it = _list_of_single_zone_images.begin(); it != _list_of_single_zone_images.end(); it++) {
		it->create_image(backend(), w);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::set_zone_width
 *
 *****************************************************************************/
bool PVParallelView::PVLinesView::set_zone_width(PVZoneID zone_id, uint32_t width)
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	
	// We want to return true if width was actually changed
	uint32_t old_width = get_zone_width(zone_id);
	uint32_t new_width = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
	bool diff = new_width != old_width;
	
	// We change the width only if it has changed...
	if (diff) {
		_zones_width[zone_id] = new_width;
	}
	
	return diff;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::translate
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::translate(int32_t view_x, uint32_t view_width, const float zoom_y)
{
	// First, set new view x (before launching anything in the future !! ;))
	
	PVZoneID previous_first_zone = set_new_view(view_x, view_width);
	if (previous_first_zone == _first_zone) {
		// "Le changement, c'est pas maintenant !"
		return;
	}

	do_translate(previous_first_zone, view_width,
	[&](PVZoneID zone_id)
	{
		assert(is_zone_drawn(zone_id));
		render_single_zone_images(zone_id, zoom_y);
	});
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::update_number_of_zones
 *
 *****************************************************************************/
int PVParallelView::PVLinesView::update_number_of_zones(int view_x, uint32_t view_width)
{
	PVCol old_zones_count = (PVCol) _zones_width.size();
	PVCol new_zones_count = get_number_of_managed_zones();
	set_nb_drawable_zones(new_zones_count);
	_zones_width.resize(new_zones_count, PVParallelView::ZoneDefaultWidth);
	// Update first zone
	set_new_view(view_x, view_width);
	return (int)new_zones_count-(int)old_zones_count;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::visit_all_zones_to_render
 *
 *****************************************************************************/
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

	const PVZoneID total_number_of_zones = get_number_of_managed_zones();
	const PVZoneID zone_index_of_first_visible_zone = get_zone_from_scene_pos(view_x);
	PVZoneID zones_to_draw = get_number_of_visible_zones();
	assert(zone_index_of_first_visible_zone >= _first_zone);
	assert(zone_index_of_first_visible_zone < _first_zone+zones_to_draw);

	left_invisible_zone = zone_index_of_first_visible_zone;

	// Process visible zones
	uint32_t current_width = 0;
	PVZoneID current_zone_index = zone_index_of_first_visible_zone;
	while (current_width < view_width && current_zone_index < total_number_of_zones && zones_to_draw > 0) {
		fzone(current_zone_index);
		const uint32_t offset = get_zone_width(current_zone_index) + PVParallelView::AxisWidth;
		current_width += offset;
		current_zone_index++;
		zones_to_draw--;
	}
	right_invisible_zone = current_zone_index;

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
		if (right_invisible_zone < total_number_of_zones) {
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






/******************************************************************************
 ******************************************************************************
 *
 * SingleZoneImages Implementation
 *
 ******************************************************************************
 *****************************************************************************/

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::cancel_all_and_wait
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::cancel_all_and_wait()
{
	// That copy is important if we are multi-threading!
	PVZoneRenderingBase_p zr = last_zr_sel;
	if (zr) {
		zr->cancel();
		zr->wait_end();
		last_zr_sel.reset();
	}

	zr = last_zr_bg;
	if (zr) {
		zr->cancel();
		zr->wait_end();
		last_zr_bg.reset();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::cancel_last_bg
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_bg()
{
	// AG: that following copy is *important* !
	PVZoneRenderingBase_p zr = last_zr_bg;
	if (zr) {
		zr->cancel();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::cancel_last_sel
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_sel()
{
	// AG: that following copy is *important* !
	PVZoneRenderingBase_p zr = last_zr_sel;
	if (zr) {
		zr->cancel();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::create_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::create_image(PVBCIDrawingBackend& backend, uint32_t width)
{
	sel = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
	bg = backend.create_image(width, PARALLELVIEW_ZT_BBITS);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::set_width
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::set_width(uint32_t width)
{
	sel->set_width(width);
	bg->set_width(width);
}

/******************************************************************************
 ******************************************************************************
 *
 * ZoneWidthWithZoomLevel Implementation
 *
 ******************************************************************************
 *****************************************************************************/


/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::decrease_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::decrease_zoom_level()
{
	if (_base_zoom_level > min_zoom_level) {
		--_base_zoom_level;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_base_zoom_level
 *
 *****************************************************************************/
int16_t PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_base_zoom_level()
{
	return _base_zoom_level;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_base_width
 *
 *****************************************************************************/
int16_t PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_base_width()
{
	return _base_width;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_width
 *
 *****************************************************************************/
uint32_t PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::get_width(int16_t global_zoom_level) const
{
	// We compute the current real zoom level
	int32_t zoom_level = PVCore::clamp((int32_t)_base_zoom_level,
	                                   min_zoom_level, max_zoom_level);

	// We compute the quotient and remainder modulo 5
	int32_t primary_zoom_level = zoom_level / zoom_divisor;  // this one for the powers of 2
	int32_t secondary_zoom_level = zoom_level % zoom_divisor;  // this one is for the powers of the 5th root of 2.

	// We compute the width without Min or Max constraints
	uint32_t brut_width = _base_width * pow(2.0, primary_zoom_level) * pow(zoom_root_value, secondary_zoom_level);

	// We clamp the value before returning anything...
	uint32_t clamped_width = PVCore::clamp(brut_width,
	                                       (uint32_t) PVParallelView::ZoneMinWidth,
	                                       (uint32_t) PVParallelView::ZoneMaxWidth);

	return clamped_width;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::increase_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::increase_zoom_level()
{
	if (_base_zoom_level <= max_zoom_level) {
		++_base_zoom_level;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_width
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_width(int16_t base_width)
{
	if ( (base_width > 15) && (base_width <2048) ) {
		_base_width = base_width;
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::set_base_zoom_level(int16_t base_zoom_level)
{
	if ( (base_zoom_level > -10000) && (base_zoom_level < 10000) ) {
		_base_zoom_level = base_zoom_level;
	}
}


