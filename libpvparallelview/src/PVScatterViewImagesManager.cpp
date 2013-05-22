/**
 * \file PVScatterViewImagesManager.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewImagesManager.h>

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
	clear_dirty_rects(data.image_all());
	data.process_bg(_data_params);
}

void PVParallelView::PVScatterViewImagesManager::process_sel()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	clear_dirty_rects(data.image_sel());
	data.process_sel(_data_params, _sel);
}

void PVParallelView::PVScatterViewImagesManager::process_all()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	clear_dirty_rects(data.image_all());
	clear_dirty_rects(data.image_sel());
	data.process_all(_data_params, _sel);
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_sel()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	return data.image_sel().get_rgb_image();
}

const QImage& PVParallelView::PVScatterViewImagesManager::get_image_all()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	return data.image_all().get_rgb_image();
}

void PVParallelView::PVScatterViewImagesManager::clear_dirty_rects(PVScatterViewImage& image) const
{
	if (!_data_params.y1_offset && !_data_params.y2_offset) {
		image.clear();
		return;
	}

	PVCore::memmove2d(
		image.get_hsv_image(),
		PVScatterViewImage::image_width,
		PVScatterViewImage::image_height,
		_data_params.map_to_view(_data_params.y1_offset),
		_data_params.map_to_view(_data_params.y2_offset)
	);

	if (_data_params.y1_offset != 0) {
		image.clear(_data_params.map_to_view(_data_params.rect_1()));
	}

	if (_data_params.y2_offset != 0) {
		image.clear(_data_params.map_to_view(_data_params.rect_2()));
	}
}
