#ifndef PVPARALLELVIEW_PVZONERENDERINGSCATTER_H
#define PVPARALLELVIEW_PVZONERENDERINGSCATTER_H

#include <pvparallelview/PVScatterViewDataInterface.h>
#include <pvparallelview/PVZoneRendering.h>

#include <functional>

namespace PVParallelView {

class PVZoneRenderingScatter: public PVZoneRenderingBase
{
	typedef PVScatterViewDataInterface::ProcessParams DataProcessParams;
	typedef std::function<void(PVScatterViewDataInterface&, DataProcessParams const&)> process_function_type;
	//typedef std::function<void(PVScatterViewDataInterface*, PVZoneID, uint64_t, uint64_t, uint64_t, uint64_t, int, double, PVCore::PVHSVColor const*)> process_function_type;

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

	PVZoneRenderingScatter(PVZoneID zid, PVScatterViewDataInterface& data_interface, DataProcessParams const& params, process_function_type const& f):
		PVZoneRenderingBase(zid),
		_data_interface(&data_interface),
		_params(params),
		_process(f)
	{ }

protected:
	//void render(std::function<void()> const& render_done = std::function<void()>());
	//inline void render(PVCore::PVHSVColor const* colors) { _process(_data_interface, get_zone_id(), _y1_min, _y1_max, _y2_min, _y2_max, _zoom, _alpha, colors); }
	inline void render() { assert(_data_interface); _process(*_data_interface, _params); }

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

}

#endif
