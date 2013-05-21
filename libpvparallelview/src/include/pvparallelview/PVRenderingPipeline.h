#ifndef PVPARALLELVIEW_PVRENDERING_PIPELINE
#define PVPARALLELVIEW_PVRENDERING_PIPELINE

#define TBB_PREVIEW_GRAPH_NODES 1
#include <tbb/flow_graph.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBuffers.h>
// The order here is important
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVRenderingPipelinePreprocessRouter.h>
#include <pvparallelview/PVZonesProcessor.h>

#include <functional>

namespace PVCore {
class PVHSVColor;
}

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVRenderingPipelinePreprocessRouter;
class PVBCIDrawingBackend;
class PVRenderingPipeline;

class PVZonesManager;

class PVRenderingPipeline: boost::noncopyable
{
	// Structures of objects passed through the graph
	struct ZoneRenderingWithBCI
	{
		// Used by TBB internally
		ZoneRenderingWithBCI() { }

		ZoneRenderingWithBCI(PVZoneRenderingBCIBase_p zr_, PVBCICodeBase* codes_, size_t ncodes_):
			zr(zr_), codes(codes_), ncodes(ncodes_)
		{ }

		PVZoneRenderingBCIBase_p zr;
		PVBCICodeBase* codes;
		size_t ncodes;
	};

	typedef PVRenderingPipelinePreprocessRouter::ZoneRenderingWithColors ZoneRenderingWithColors;

	// Ports type
	typedef tbb::flow::receiver<ZoneRenderingWithColors> input_port_zrc_type;
	typedef tbb::flow::receiver<PVZoneRenderingBase_p> input_port_cancel_type;

	// Process nodes structures
	struct Preprocessor: boost::noncopyable
	{
		typedef std::function<void(PVZoneID)> preprocess_func_type;

		typedef PVRenderingPipelinePreprocessRouter::process_or_type process_or_type;
		typedef PVRenderingPipelinePreprocessRouter::multinode_router multinode_router;

		typedef tbb::flow::receiver<PVZoneRenderingBase_p> input_port_type;

		Preprocessor(tbb::flow::graph& g, input_port_zrc_type& node_in_job, input_port_cancel_type& node_cancel_job, preprocess_func_type const& f, PVCore::PVHSVColor const* colors, size_t nzones);

		inline input_port_type& input_port() { return tbb::flow::input_port<PVRenderingPipelinePreprocessRouter::InputIdxDirect>(node_or); }

		PVRenderingPipelinePreprocessRouter router;
		tbb::flow::function_node<PVZoneRenderingBase_p, PVZoneRenderingBase_p> node_process;
		process_or_type node_or;
		multinode_router node_router;
	};

	struct DirectInput: boost::noncopyable
	{
		typedef tbb::flow::multifunction_node<PVZoneRenderingBase_p, std::tuple<ZoneRenderingWithColors, PVZoneRenderingBase_p>> direct_process_type;
		DirectInput(tbb::flow::graph& g, input_port_zrc_type& node_in_job, input_port_cancel_type& node_cancel_job, PVCore::PVHSVColor const* colors_);

		direct_process_type node_process;
	};

	// Cancellation points types
	constexpr static size_t cp_continue_port = 0;
	constexpr static size_t cp_cancel_port = 1;

	// Workflow router points types
	constexpr static size_t wr_bci = 0;
	constexpr static size_t wr_scatter = 1;

	typedef tbb::flow::multifunction_node<ZoneRenderingWithColors, std::tuple<ZoneRenderingWithColors, PVZoneRenderingBase_p, tbb::flow::continue_msg> > cp_postlimiter_type;
	typedef tbb::flow::multifunction_node<ZoneRenderingWithBCI, std::tuple<ZoneRenderingWithBCI, ZoneRenderingWithBCI> > cp_postcomputebci_type;
	typedef tbb::flow::multifunction_node<ZoneRenderingWithColors, std::tuple<ZoneRenderingWithColors, ZoneRenderingWithColors> > workflow_router_type;

	friend class Preprocess;
	friend class DirectInput;

public:
	typedef PVRenderingPipelinePreprocessRouter::ZoneState ZoneState;
	typedef Preprocessor::preprocess_func_type preprocess_func_type;

public:
	PVRenderingPipeline(PVBCIDrawingBackend& bci_backend);
	~PVRenderingPipeline();

public:
	PVZonesProcessor declare_processor(preprocess_func_type const& f, PVCore::PVHSVColor const* colors, size_t nzones);
	PVZonesProcessor declare_processor(PVCore::PVHSVColor const* colors);

	void cancel_all();
	void wait_for_all();

public:
	/*
	template <size_t bbits>
	static void* allocate_zr()
	{
		return malloc(sizeof(PVZoneRendering<bbits>));
	}
	static void free_zr(PVZoneRenderingBase_p zr);*/

private:
	inline tbb::flow::graph& tbb_graph() { return _g; }
	inline tbb::flow::graph const& tbb_graph() const { return _g; }

private:
	std::vector<Preprocessor*> _preprocessors;
	std::vector<DirectInput*> _direct_inputs;
	PVBCIBuffers<BCI_BUFFERS_COUNT> _bci_buffers;

private:
	tbb::flow::graph _g;
	tbb::flow::function_node<ZoneRenderingWithColors, ZoneRenderingWithBCI>* _node_compute_bci;
	tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>* _node_draw_bci;
	tbb::flow::function_node<ZoneRenderingWithBCI, PVZoneRenderingBase_p>* _node_cleanup_bci;
	tbb::flow::function_node<PVZoneRenderingBase_p>* _node_finish;
	tbb::flow::function_node<ZoneRenderingWithColors, PVZoneRenderingBase_p>* _node_compute_scatter;
	workflow_router_type* _workflow_router;
	

	// Cancellation points
	cp_postlimiter_type* _cp_postlimiter;
	cp_postcomputebci_type* _cp_postcomputebci;

	tbb::flow::limiter_node<ZoneRenderingWithColors> _node_limiter;
	tbb::flow::buffer_node<ZoneRenderingWithColors> _node_buffer;
};

}

#endif
