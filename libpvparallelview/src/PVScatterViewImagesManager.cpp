/**
 * \file PVScatterViewImagesManager.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewImagesManager.h>
#include <pvparallelview/PVZoneRenderingScatter.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVZonesProcessor.h>

PVParallelView::PVScatterViewImagesManager::PVScatterViewImagesManager(
	PVZoneID const zid,
	PVZonesProcessor& zp_bg,
	PVZonesProcessor& zp_sel,
	PVZonesManager const& zm,
	const PVCore::PVHSVColor* colors,
	Picviz::PVSelection const& sel
):
	_zid(zid),
	_zm(zm),
	_sel(sel),
	_data_params(zm.get_zone_tree<PVParallelView::PVZoomedZoneTree>(zid), colors),
	_zp_bg(zp_bg),
	_zp_sel(zp_sel),
	_img_update_receiver(nullptr)
{
}

void PVParallelView::PVScatterViewImagesManager::cancel_all_and_wait()
{
	if (_zr_bg) {
		_zr_bg->cancel();
	}
	if (_zr_sel) {
		_zr_sel->cancel();
	}
	if (_zr_bg) {
		_zr_bg->wait_end();
	}
	if (_zr_sel) {
		_zr_sel->wait_end();
	}
}

void PVParallelView::PVScatterViewImagesManager::set_zone(PVZoneID const zid)
{
	cancel_all_and_wait();
	_data_params.zzt = &get_zones_manager().get_zone_tree<PVParallelView::PVZoomedZoneTree>(zid);
	_zid = zid;
}

bool PVParallelView::PVScatterViewImagesManager::change_and_process_view(
	const uint64_t y1_min,
	const uint64_t y1_max,
	const uint64_t y2_min,
	const uint64_t y2_max,
	const int zoom,
	const double alpha
)
{
	if (!params_changed(y1_min, y1_max, y2_min, y2_max, zoom, alpha)) {
		return false;
	}

	set_params(y1_min, y1_max, y2_min, y2_max, zoom, alpha);

	process_all();

	return true;
}

void PVParallelView::PVScatterViewImagesManager::set_params(
	uint64_t y1_min,
	uint64_t y1_max,
	uint64_t y2_min,
	uint64_t y2_max,
	int zoom,
	double alpha
)
{
	// Translation
	if (zoom == last_zoom() && alpha == last_alpha()) {
		_data_params.y1_offset = _data_params.y1_min - y1_min;
		_data_params.y2_offset = _data_params.y2_min - y2_min;
	}
	else {
		_data_params.y1_offset = 0;
		_data_params.y2_offset = 0;
	}
	_data_params.y1_min = y1_min;
	_data_params.y1_max = y1_max;
	_data_params.y2_min = y2_min;
	_data_params.y2_max = y2_max;
	_data_params.zoom = zoom;
	_data_params.alpha = alpha;
}

void PVParallelView::PVScatterViewImagesManager::process_bg()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;

	PVZoneRenderingScatter_p zr(new PVZoneRenderingScatter(_zid, data, _data_params,
			[&](PVScatterViewDataInterface& data_if, DataProcessParams const& params, tbb::task_group_context& ctxt)
			{
				PVParallelView::PVScatterViewImagesManager::clear_dirty_rects(params, data_if.image_bg());
				data_if.process_bg(params, &ctxt);
			}));

	connect_zr(*zr, "update_img_bg");

	PVZoneRenderingScatter_p old_zr = _zr_bg;
	_zr_bg = zr;
	if (old_zr) {
		//old_zr->cancel_and_add_job(_zp_bg, zr);
		old_zr->wait_end();
	}
	else {
		_zp_bg.add_job(zr);
	}
}

void PVParallelView::PVScatterViewImagesManager::process_sel()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;

	PVZoneRenderingScatter_p zr(new PVZoneRenderingScatter(_zid, data, _data_params,
			[&](PVScatterViewDataInterface& data_if, DataProcessParams const& params, tbb::task_group_context& ctxt)
			{
				PVScatterViewImagesManager::clear_dirty_rects(params, data_if.image_sel());
				data_if.process_sel(params, this->_sel, &ctxt);
			}));

	connect_zr(*zr, "update_img_sel");

	PVZoneRenderingScatter_p old_zr = _zr_sel;
	_zr_sel = zr;
	if (old_zr) {
		old_zr->cancel_and_add_job(_zp_sel, zr);
	}
	else {
		_zp_sel.add_job(zr);
	}
}

void PVParallelView::PVScatterViewImagesManager::process_all()
{
	// AG: TODO: scheduling here isn't trivial, because we should be able to
	// say: "launch the process_all job when both _zr_sel and _zr_bg have
	// finished". Moreover, this implies for process_bg and process_sel to be
	// able to cancel both "_zr_all" and _zr_sel, with the same issue !
	// For now, just launch the selection and background job.
	/*PVScatterViewData& data = full_view() ? _data_z0 : _data;
	clear_dirty_rects(data.image_all());
	clear_dirty_rects(data.image_sel());
	data.process_all(_data_params, _sel);*/
	process_sel();
	process_bg();
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_sel() const
{
	PVScatterViewData const& data = full_view() ? _data_z0 : _data;
	return data.image_sel().get_rgb_image();
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_all() const
{
	PVScatterViewData const& data = full_view() ? _data_z0 : _data;
	return data.image_bg().get_rgb_image();
}

void PVParallelView::PVScatterViewImagesManager::connect_zr(PVZoneRenderingScatter& zr, const char* slot)
{
	if (_img_update_receiver) {
		zr.set_render_finished_slot(_img_update_receiver, slot);
	}
}

void PVParallelView::PVScatterViewImagesManager::clear_dirty_rects(DataProcessParams const& params, PVScatterViewImage& image)
{
	if (!params.y1_offset && !params.y2_offset) {
		image.clear();
		return;
	}

	PVCore::memmove2d(
		image.get_hsv_image(),
		PVScatterViewImage::image_width,
		PVScatterViewImage::image_height,
		params.map_to_view(params.y1_offset),
		params.map_to_view(params.y2_offset)
	);

	if (params.y1_offset != 0) {
		image.clear(params.map_to_view(params.rect_1()));
	}

	if (params.y2_offset != 0) {
		image.clear(params.map_to_view(params.rect_2()));
	}
}
