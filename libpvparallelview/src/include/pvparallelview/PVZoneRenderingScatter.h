/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZONERENDERINGSCATTER_H
#define PVPARALLELVIEW_PVZONERENDERINGSCATTER_H

#include <pvparallelview/PVScatterViewDataInterface.h>
#include <pvparallelview/PVZoneRenderingTBB.h>

#include <functional>

namespace PVParallelView
{

class PVZoneRenderingScatter : public PVZoneRenderingTBB
{
	typedef PVScatterViewDataInterface::ProcessParams DataProcessParams;
	typedef std::function<void(
	    PVScatterViewDataInterface&, DataProcessParams const&, tbb::task_group_context&)>
	    process_function_type;
	// typedef std::function<void(PVScatterViewDataInterface*, PVZoneID, uint64_t, uint64_t,
	// uint64_t, uint64_t, int, double, PVCore::PVHSVColor const*)> process_function_type;

	friend class PVRenderingPipeline;

  public:
	/*PVZoneRenderingScatter(PVZoneID zid,
	                       uint64_t y1_min,
	                       uint64_t y1_max,
	                       uint64_t y2_min,
	                       uint64_t y2_max,
	                       int zoom,
	                       double alpha,
	                       process_function_type const& f):
	    PVZoneRenderingBase(zid),
	    _y1_min(y1_min),
	    _y1_max(y1_max),
	    _y2_min(y2_min),
	    _y2_max(y2_max),
	    _zoom(zoom),
	    _alpha(alpha),
	    _process(f)
	{
	}*/

	PVZoneRenderingScatter(PVZoneID zid,
	                       PVScatterViewDataInterface& data_interface,
	                       DataProcessParams params,
	                       process_function_type f)
	    : PVZoneRenderingTBB(zid)
	    , _data_interface(&data_interface)
	    , _params(std::move(params))
	    , _process(std::move(f))
	{
	}

  protected:
	// void render(std::function<void()> const& render_done = std::function<void()>());
	// inline void render(PVCore::PVHSVColor const* colors) { _process(_data_interface,
	// get_zone_id(), _y1_min, _y1_max, _y2_min, _y2_max, _zoom, _alpha, colors); }
	inline void render()
	{
		assert(_data_interface);
		_process(*_data_interface, _params, get_task_group_context());
	}

  private:
	/*
	uint64_t _y1_min;
	uint64_t _y1_max;
	uint64_t _y2_min;
	uint64_t _y2_max;
	int _zoom;
	double _alpha;*/

	PVScatterViewDataInterface* _data_interface;
	DataProcessParams _params;

	process_function_type _process;
};
} // namespace PVParallelView

#endif
