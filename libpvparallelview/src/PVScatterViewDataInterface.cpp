/**
 * \file PVScatterViewDataInterface.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <cassert>

#include <pvparallelview/common.h>
#include <pvparallelview/PVScatterViewDataInterface.h>

#include <tbb/task.h>

typedef PVParallelView::PVScatterViewDataInterface::ProcessParams ProcessParamsImpl;

ProcessParamsImpl::dirty_rect ProcessParamsImpl::rect_1() const
{
	ProcessParamsImpl::dirty_rect drect;

	drect.y1_min = y1_offset > 0 ? y1_min : y1_max + (y1_offset+1);
	drect.y1_max = y1_offset > 0 ? y1_min + (y1_offset-1) : y1_max;
	drect.y2_min = y2_offset < 0 ? y2_min : y2_min + (y2_offset+1);
	drect.y2_max = y2_offset < 0 ? y2_max + (y2_offset+1) : y2_max;

	assert(drect.y1_min >= y1_min);
	assert(drect.y1_max <= y1_max);
	assert(drect.y2_min >= y2_min);
	assert(drect.y2_max <= y2_max);

	return drect;
}

ProcessParamsImpl::dirty_rect ProcessParamsImpl::rect_2() const
{
	dirty_rect drect;

	drect.y1_min = y1_min;
	drect.y1_max = y1_max;
	drect.y2_min = y2_offset > 0 ? y2_min : y2_max + (y2_offset+1);
	drect.y2_max = y2_offset > 0 ? y2_min + (y2_offset-1) : y2_max;

	assert(drect.y1_min >= y1_min);
	assert(drect.y1_max <= y1_max);
	assert(drect.y2_min >= y2_min);
	assert(drect.y2_max <= y2_max);

	return drect;
}

int32_t ProcessParamsImpl::map_to_view(int64_t scene_value) const
{
	return ((int64_t)(ceil(scene_value * alpha))) >> (32 - PARALLELVIEW_ZT_BBITS - zoom);
}

QRect ProcessParamsImpl::map_to_view(const dirty_rect& rect) const
{
	return QRect(
		QPoint(map_to_view(rect.y1_min - y1_min), map_to_view(rect.y2_min - y2_min)),
		QPoint(map_to_view(rect.y1_max - y1_min), map_to_view(rect.y2_max - y2_min))
	);
}

bool PVParallelView::PVScatterViewDataInterface::is_ctxt_cancelled(tbb::task_group_context* ctxt)
{
	return (ctxt && ctxt->is_group_execution_cancelled());
}

bool ProcessParamsImpl::params_changed(
		uint64_t y1_min_,
		uint64_t y1_max_,
		uint64_t y2_min_,
		uint64_t y2_max_,
		int zoom_,
		double alpha_) const
{
	return !(y1_min_ == y1_min &&
			y1_max_ == y1_max &&
			y2_min_ == y2_min &&
			y2_max_ == y2_max &&
			zoom_ == zoom &&
			alpha_ == alpha);
}

void ProcessParamsImpl::set_params(
		uint64_t y1_min_,
		uint64_t y1_max_,
		uint64_t y2_min_,
		uint64_t y2_max_,
		int zoom_,
		double alpha_)
{
	// Translation
	if (zoom_ == zoom && alpha_ == alpha &&
		(y1_max-y1_min) == (y1_max_ - y1_min_) &&
		(y2_max-y2_min) == (y2_max_ - y2_min_)) {
		y1_offset = y1_min - y1_min_;
		y2_offset = y2_min - y2_min_;
	}
	else {
		y1_offset = 0;
		y2_offset = 0;
	}
	y1_min = y1_min_;
	y1_max = y1_max_;
	y2_min = y2_min_;
	y2_max = y2_max_;
	zoom = zoom_;
	alpha = alpha_;
}
