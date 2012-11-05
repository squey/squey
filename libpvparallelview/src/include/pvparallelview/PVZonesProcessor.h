#ifndef PVPARALLELVIEW_PVZONESPROCESSOR_H
#define PVPARALLELVIEW_PVZONESPROCESSOR_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneRendering_types.h>

#define TBB_PREVIEW_GRAPH_NODES 1
#include <tbb/flow_graph.h>

namespace PVCore {
class PVHSVColor;
}

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVRenderingPipeline;
class PVRenderingPipelinePreprocessRouter;
class PVZonesManager;

class PVZonesProcessor
{
	friend class PVRenderingPipeline;
	typedef tbb::flow::receiver<PVZoneRenderingBase_p> receiver_type;

protected:
	PVZonesProcessor(receiver_type& in_port, PVRenderingPipelinePreprocessRouter* preprocess = nullptr):
		_in_port(&in_port), _preprocess(preprocess)
	{ }

public:
	PVZonesProcessor():
		_in_port(nullptr),
		_preprocess(nullptr)
	{ }

	PVZonesProcessor(PVZonesProcessor const& zp):
		_in_port(zp._in_port), _preprocess(zp._preprocess)
	{ }

	inline PVZonesProcessor& operator=(PVZonesProcessor const& zp)
	{
		if (&zp != this) {
			_in_port = zp._in_port;
			_preprocess = zp._preprocess;
		}
		return *this;
	}

public:
	inline bool add_job(PVZoneRenderingBase_p const& zr)
	{
		assert(_in_port);
		return _in_port->try_put(zr);
	}

	// Preprocess router specific functions
	void set_number_zones(const PVZoneID n);
	void invalidate_zone_preprocessing(const PVZoneID zone_id);

public:
	// Static helper functions, implemented in PVZonesProcessor.cpp
	static PVZonesProcessor declare_processor_zm_sel(PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors, Picviz::PVSelection const& sel);
	static PVZonesProcessor declare_processor_direct(PVRenderingPipeline& pipeline, PVCore::PVHSVColor const* colors);

private:
	receiver_type* _in_port;
	PVRenderingPipelinePreprocessRouter* _preprocess;
};

}

#endif
