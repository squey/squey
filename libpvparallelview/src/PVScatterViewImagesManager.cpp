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

void PVParallelView::PVScatterViewImagesManager::process_bg()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;

	PVZoneRenderingScatter_p zr(new PVZoneRenderingScatter(_zid, data, _data_params,
			[](PVScatterViewDataInterface& data_if, DataProcessParams const& params, tbb::task_group_context& ctxt)
			{
				data_if.image_bg().clear();
				data_if.process_bg(params, &ctxt);
			}));

	connect_zr(*zr, "update_img_bg");

	PVZoneRenderingScatter_p old_zr = _zr_bg;
	_zr_bg = zr;
	if (old_zr) {
		old_zr->cancel_and_add_job(_zp_bg, zr);
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
				data_if.image_sel().clear();
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
	/*PVScatterViewData& data = full_view() ? _data_z0 : _data;
	data.image_all().clear();
	data.image_sel().clear();
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
