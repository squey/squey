//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
    Squey::PVSelection const& sel)
    : _zm(zm)
    , _sel(sel)
    , _colors(colors)
    , _zp_bg(zp_bg)
    , _zp_sel(zp_sel)
    , _img_update_receiver(nullptr)
{
	set_zone(zid);
}

PVParallelView::PVScatterViewImagesManager::~PVScatterViewImagesManager()
{
	cancel_all_and_wait();
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
	_zid = zid;

	PVParallelView::PVZoomedZoneTree const& zzt = get_zones_manager().get_zoom_zone_tree(_zid);
	_data.set_zoomed_zone_tree(zzt);
}

bool PVParallelView::PVScatterViewImagesManager::change_and_process_view(const uint64_t y1_min,
                                                                         const uint64_t y1_max,
                                                                         const uint64_t y2_min,
                                                                         const uint64_t y2_max,
                                                                         const int zoom,
                                                                         const double alpha)
{
	DataProcessParams params;
	bool ret = false;
	PVScatterViewData& data = _data;

	params = data.image_bg_process_params();
	params.colors = _colors;
	if (params.params_changed(y1_min, y1_max, y2_min, y2_max, zoom, alpha)) {
		params.set_params(y1_min, y1_max, y2_min, y2_max, zoom, alpha);
		process_bg(params);
		ret = true;
	}

	params = data.image_sel_process_params();
	params.colors = _colors;
	if (params.params_changed(y1_min, y1_max, y2_min, y2_max, zoom, alpha)) {
		params.set_params(y1_min, y1_max, y2_min, y2_max, zoom, alpha);
		process_sel(params);
		ret = true;
	}

	return ret;
}

void PVParallelView::PVScatterViewImagesManager::process_bg()
{
	PVScatterViewData& data = _data;
	process_bg(data.image_bg_process_params());
}

void PVParallelView::PVScatterViewImagesManager::process_sel()
{
	PVScatterViewData& data = _data;
	process_sel(data.image_sel_process_params());
}

void PVParallelView::PVScatterViewImagesManager::process_bg(DataProcessParams const& process_params)
{
	if (_zid.is_invalid()) {
		return;
	}

	PVScatterViewData& data = _data;

	PVZoneRenderingScatter_p zr(new PVZoneRenderingScatter(
	    _zid, data, process_params,
	    [](PVScatterViewDataInterface& data_if, DataProcessParams const& params,
	        tbb::task_group_context& ctxt) {
		    PVScatterViewImagesManager::copy_processed_in_processing(
		        params, data_if.image_bg_processing(), data_if.image_bg());
		    data_if.process_image_bg(params, &ctxt);
		}));

	connect_zr(*zr, "update_img_bg");

	// if (_zr_bg) {
	// 	_zr_bg->cancel();
	// }
	// _zr_bg = zr;
	// _zp_bg.add_job(zr);
	// The code below is black magic, so if it breaks, the code above is fine.

	PVZoneRenderingScatter_p old_zr = _zr_bg;
	_zr_bg = zr;
	if (old_zr) {
		old_zr->cancel_and_add_job(_zp_bg, zr);
	} else {
		_zp_bg.add_job(zr);
	}
}

void PVParallelView::PVScatterViewImagesManager::process_sel(
    DataProcessParams const& process_params)
{
	if (_zid.is_invalid()) {
		return;
	}

	PVScatterViewData& data = _data;

	PVZoneRenderingScatter_p zr(new PVZoneRenderingScatter(
	    _zid, data, process_params,
	    [this](PVScatterViewDataInterface& data_if, DataProcessParams const& params,
	        tbb::task_group_context& ctxt) {
		    PVScatterViewImagesManager::copy_processed_in_processing(
		        params, data_if.image_sel_processing(), data_if.image_sel());
		    data_if.process_image_sel(params, this->_sel, &ctxt);
		}));

	connect_zr(*zr, "update_img_sel");

	// if (_zr_sel) {
	// 	_zr_sel->cancel();
	// }
	// _zr_sel = zr;
	// _zp_sel.add_job(zr);
	// The code below is black magic, so if it breaks, the code above is fine.

	PVZoneRenderingScatter_p old_zr = _zr_sel;
	_zr_sel = zr;
	if (old_zr) {
		old_zr->cancel_and_add_job(_zp_sel, zr);
	} else {
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

	cancel_all_and_wait();

	PVParallelView::PVZoomedZoneTree const& zzt = get_zones_manager().get_zoom_zone_tree(_zid);
	_data.set_zoomed_zone_tree(zzt);

	process_sel();
	process_bg();
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_sel() const
{
	PVScatterViewData const& data = _data;
	return data.image_sel().get_rgb_image();
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_all() const
{
	PVScatterViewData const& data = _data;
	return data.image_bg().get_rgb_image();
}

void PVParallelView::PVScatterViewImagesManager::connect_zr(PVZoneRenderingScatter& zr,
                                                            const char* slot)
{
	if (_img_update_receiver) {
		zr.set_render_finished_slot(_img_update_receiver, slot);
	}
}

void PVParallelView::PVScatterViewImagesManager::copy_processed_in_processing(
    DataProcessParams const& params,
    PVScatterViewImage& processing,
    PVScatterViewImage const& processed)
{
	processing.clear();

	if (params.can_optimize_translation()) {
		PVCore::memcpy2d(processing.get_hsv_image(), processed.get_hsv_image(),
		                 PVScatterViewImage::image_width, PVScatterViewImage::image_height,
		                 params.map_to_view(params.y1_offset),
		                 params.map_to_view(params.y2_offset));
	}
}
