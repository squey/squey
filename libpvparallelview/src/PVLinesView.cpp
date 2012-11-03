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

/******************************************************************************
 *
 * PVParallelView::PVLinesView::PVLinesView
 *
 *****************************************************************************/
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
	set_nb_drawable_zones(get_number_of_managed_zones());


	_zones_width.resize(get_number_of_managed_zones(), PVParallelView::ZoneDefaultWidth);
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
			Q_ARG(void*, NULL),
			Q_ARG(PVZoneID, zone_id));
	QMetaObject::invokeMethod(_img_update_receiver, "zr_bg_finished",  Qt::QueuedConnection,
			Q_ARG(void*, NULL),
			Q_ARG(PVZoneID, zone_id));
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
 * PVParallelView::PVLinesView::do_translate
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::do_translate(PVZoneID previous_first_zone, uint32_t view_width, std::function<void(PVZoneID)> fzone_draw)
{
	const PVZoneID nzones_img = _list_of_single_zone_images.size();
	const PVZoneID diff = std::abs(_first_zone - previous_first_zone);
	if (diff >= nzones_img) {
		visit_all_zones_to_render(view_width, fzone_draw);
		return;
	}

	if (_first_zone > previous_first_zone) {
		// Translation to the left

		const PVZoneID n = diff;
		left_shift_images(n);

		const PVZoneID nimgs = _list_of_single_zone_images.size();
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
		// Translation to the right

		right_shift_images(diff);
		const PVZoneID n = diff;
		PVZoneID first_z_to_render = _first_zone;
		const PVZoneID last_z = picviz_min(_first_zone + n, get_number_of_managed_zones());

		// If a rendering job is provided, tell him that we virtually have rendered from last_z to get_last_drawn_zone()
		/*if (job) {
			for (PVZoneID z = last_z; z <= get_last_drawn_zone(); z++) {
				job->zone_finished(z);
			}
		}*/
		if (_img_update_receiver) {
			for (PVZoneID zone_id = last_z; zone_id <= get_last_drawn_zone(); zone_id++) {
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
 * PVParallelView::PVLinesView::get_first_zone_from_viewport
 *
 *****************************************************************************/
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

	const PVZoneID total_number_of_zones = get_number_of_managed_zones();
	const PVZoneID zfirst_visible = get_zone_from_scene_pos(view_x);
	int zones_drawable = _list_of_single_zone_images.size();

	uint32_t cur_width = 0;
	PVZoneID cur_z = zfirst_visible;
	while (cur_width < view_width && cur_z < total_number_of_zones && zones_drawable > 0) {
		const uint32_t offset = get_zone_width(cur_z) + PVParallelView::AxisWidth;
		cur_width += offset;
		cur_z++;
		zones_drawable--;
	}

	if (cur_z >= total_number_of_zones) {
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
	ret = PVCore::clamp(ret, 0, (PVZoneID) std::max(0, (total_number_of_zones-(PVZoneID)_list_of_single_zone_images.size())));

	return ret;
 }

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_image_index_of_zone
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_image_index_of_zone(PVZoneID zone_id) const
{
	return (is_zone_drawn(zone_id)) ? zone_id-get_first_drawn_zone() : PVZONEID_INVALID;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_number_of_managed_zones
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_number_of_managed_zones() const
{
	return get_zones_manager().get_number_of_zones();
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_zone_absolute_pos
 *
 *****************************************************************************/
uint32_t PVParallelView::PVLinesView::get_zone_absolute_pos(PVZoneID zone_id) const
{
	assert(zone_id < (PVZoneID) _zones_width.size());
	uint32_t pos = 0;
	for (PVZoneID z = 0; z < zone_id; z++) {
		pos += _zones_width[z] + PVParallelView::AxisWidth;
	}
	return pos;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::get_zone_from_scene_pos
 *
 *****************************************************************************/
PVZoneID PVParallelView::PVLinesView::get_zone_from_scene_pos(int abs_pos) const
{
	PVZoneID zone_id = 0;
	ssize_t pos = 0;
	for (; zone_id < (PVZoneID) (_zones_width.size()-1) ; zone_id++)
	{
		pos += _zones_width[zone_id] + PVParallelView::AxisWidth;
		if (pos > abs_pos) {
			break;
		}
	}

	assert(zone_id < (PVZoneID) _zones_width.size());
	return zone_id;
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::left_shift_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::left_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _list_of_single_zone_images.size());
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

	SingleZoneImages& single_zone_images = get_single_zone_images(zone_id);
	single_zone_images.cancel_last_bg();
	const uint32_t width = get_zone_width(zone_id);
	single_zone_images.bg->set_width(width);

	PVZoneRendering<PARALLELVIEW_ZT_BBITS>* zr = new (PVRenderingPipeline::allocate_zr<PARALLELVIEW_ZT_BBITS>()) PVZoneRendering<PARALLELVIEW_ZT_BBITS>(zone_id,
		[&,width,zoom_y](PVZoneID zone_id, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(zone_id).browse_tree_bci(colors, codes);
		},
		*single_zone_images.bg,
		0,
		width,
		zoom_y,
		false // not reversed
		);

	connect_zr(zr, "zr_bg_finished");
	single_zone_images.last_zr_bg = zr;

	_processor_bg.add_job(*zr);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::render_single_zone_sel_image
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::render_single_zone_sel_image(PVZoneID zone_id, const float zoom_y)
{
	assert(is_zone_drawn(zone_id));

	SingleZoneImages& single_zone_images = get_single_zone_images(zone_id);
	single_zone_images.cancel_last_sel();
	const uint32_t width = get_zone_width(zone_id);
	single_zone_images.sel->set_width(width);

	PVZoneRendering<PARALLELVIEW_ZT_BBITS>* zr = new (PVRenderingPipeline::allocate_zr<PARALLELVIEW_ZT_BBITS>()) PVZoneRendering<PARALLELVIEW_ZT_BBITS>(zone_id,
		[&,width,zoom_y](PVZoneID zone_id, PVCore::PVHSVColor const* colors, PVBCICode<PARALLELVIEW_ZT_BBITS>* codes)
		{
			return this->get_zones_manager().get_zone_tree<PVZoneTree>(zone_id).browse_tree_bci_sel(colors, codes);
		},
		*single_zone_images.sel,
		0,
		width,
		zoom_y,
		false // not reversed
		);

	connect_zr(zr, "zr_sel_finished");
	single_zone_images.last_zr_sel = zr;

	_processor_sel.add_job(*zr);
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::right_shift_images
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::right_shift_images(PVZoneID s)
{
	assert(s < (PVZoneID) _list_of_single_zone_images.size());
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
	PVZoneID old_nzones = _list_of_single_zone_images.size();
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
	// Returns true if width was actual changed
	uint32_t old_width = get_zone_width(zone_id);
	uint32_t new_width = PVCore::clamp(width, (uint32_t) PVParallelView::ZoneMinWidth, (uint32_t) PVParallelView::ZoneMaxWidth);
	bool diff = new_width != old_width;
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
	const PVZoneID zfirst_visible = get_zone_from_scene_pos(view_x);
	PVZoneID zones_to_draw = _list_of_single_zone_images.size();
	assert(zfirst_visible >= _first_zone);
	assert(zfirst_visible < _first_zone+zones_to_draw);

	left_invisible_zone = zfirst_visible;

	// Process visible zones
	uint32_t cur_width = 0;
	PVZoneID cur_z = zfirst_visible;
	while (cur_width < view_width && cur_z < total_number_of_zones && zones_to_draw > 0) {
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
	if (last_zr_sel) {
		last_zr_sel->cancel();
		last_zr_sel->wait_end();
		PVZoneRenderingBase* zr = last_zr_sel;
		last_zr_sel = nullptr;
		PVRenderingPipeline::free_zr(zr);
	}

	//FIXME : PhS : Why are these two codes different ???
	if (last_zr_bg) {
		last_zr_bg->cancel();
		last_zr_bg->wait_end();
		last_zr_bg = nullptr;
		PVZoneRenderingBase* zr = last_zr_bg;
		PVRenderingPipeline::free_zr(zr);
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::cancel_last_bg
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_bg()
{
	if (last_zr_bg) {
		last_zr_bg->cancel();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVLinesView::SingleZoneImages::cancel_last_sel
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::SingleZoneImages::cancel_last_sel()
{
	if (last_zr_sel) {
		last_zr_sel->cancel();
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
	if (_base_zoom_level > -10000) {
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
 * PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::increase_zoom_level
 *
 *****************************************************************************/
void PVParallelView::PVLinesView::ZoneWidthWithZoomLevel::increase_zoom_level()
{
	if (_base_zoom_level < 10000) {
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


