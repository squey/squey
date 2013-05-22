/**
 * \file PVScatterViewDataInterface.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

#include <pvparallelview/PVScatterViewDataInterface.h>

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
	return (int64_t)((scene_value) * alpha) >> (32 - PARALLELVIEW_ZT_BBITS - zoom);
}

QRect ProcessParamsImpl::map_to_view(const dirty_rect& rect) const
{
	return QRect(
		QPoint(map_to_view(rect.y1_min - y1_min), map_to_view(rect.y2_min - y2_min)),
		QPoint(map_to_view(rect.y1_max - y1_min), map_to_view(rect.y2_max - y2_min))
	);
}
