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

#include <cassert>
#include <math.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVScatterViewDataInterface.h>

#include <tbb/task.h>

using ProcessParamsImpl = PVParallelView::PVScatterViewDataInterface::ProcessParams;

ProcessParamsImpl::dirty_rect ProcessParamsImpl::rect_1() const
{
	ProcessParamsImpl::dirty_rect drect;

	drect.y1_min = y1_offset > 0 ? y1_min : y1_max + (y1_offset + 1);
	drect.y1_max = y1_offset > 0 ? y1_min + (y1_offset - 1) : y1_max;
	drect.y2_min = y2_offset < 0 ? y2_min : y2_min + (y2_offset + 1);
	drect.y2_max = y2_offset < 0 ? y2_max + (y2_offset + 1) : y2_max;

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
	drect.y2_min = y2_offset > 0 ? y2_min : y2_max + (y2_offset + 1);
	drect.y2_max = y2_offset > 0 ? y2_min + (y2_offset - 1) : y2_max;

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
	return QRect(QPoint(map_to_view(rect.y1_min - y1_min), map_to_view(rect.y2_min - y2_min)),
	             QPoint(map_to_view(rect.y1_max - y1_min), map_to_view(rect.y2_max - y2_min)));
}

bool PVParallelView::PVScatterViewDataInterface::is_ctxt_cancelled(tbb::task_group_context* ctxt)
{
	return (ctxt && ctxt->is_group_execution_cancelled());
}

bool ProcessParamsImpl::params_changed(uint64_t y1_min_,
                                       uint64_t y1_max_,
                                       uint64_t y2_min_,
                                       uint64_t y2_max_,
                                       int zoom_,
                                       double alpha_) const
{
	return !(y1_min_ == y1_min && y1_max_ == y1_max && y2_min_ == y2_min && y2_max_ == y2_max &&
	         zoom_ == zoom && alpha_ == alpha);
}

bool PVParallelView::PVScatterViewDataInterface::ProcessParams::can_optimize_translation() const
{
	return (y1_offset != 0 || y2_offset != 0) &&
	       (static_cast<uint64_t>(std::abs(y1_offset)) < (y1_max - y1_min)) &&
	       (static_cast<uint64_t>(std::abs(y2_offset)) < (y2_max - y2_min));
}

void ProcessParamsImpl::set_params(uint64_t y1_min_,
                                   uint64_t y1_max_,
                                   uint64_t y2_min_,
                                   uint64_t y2_max_,
                                   int zoom_,
                                   double alpha_)
{
	// Translation
	if (zoom_ == zoom && alpha_ == alpha && (y1_max - y1_min) == (y1_max_ - y1_min_) &&
	    (y2_max - y2_min) == (y2_max_ - y2_min_)) {
		y1_offset = y1_min - y1_min_;
		y2_offset = y2_min - y2_min_;
	} else {
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
