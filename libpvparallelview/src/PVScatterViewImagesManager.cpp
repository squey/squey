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
	if (y1_min == _data_params.y1_min &&
		y1_max == _data_params.y1_max &&
		y2_min == _data_params.y2_min &&
		y2_max == _data_params.y2_max &&
		zoom == _data_params.zoom &&
		alpha == _data_params.alpha) {
		return false;
	}

	_data_params.y1_min = y1_min;
	_data_params.y1_max = y1_max;
	_data_params.y2_min = y2_min;
	_data_params.y2_max = y2_max;
	_data_params.zoom = zoom;
	_data_params.alpha = alpha;

	process_all();

	return true;
}

void PVParallelView::PVScatterViewImagesManager::process_bg()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	data.image_all().clear();
	data.process_bg(_data_params);
}

void PVParallelView::PVScatterViewImagesManager::process_sel()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	data.image_all().clear();
	data.process_sel(_data_params, _sel);
}

void PVParallelView::PVScatterViewImagesManager::process_all()
{
	PVScatterViewData& data = full_view() ? _data_z0 : _data;
	data.image_all().clear();
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
