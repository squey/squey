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

#ifndef PVPARALLELVIEW_PVRENDERING_PIPELINE
#define PVPARALLELVIEW_PVRENDERING_PIPELINE

#define TBB_PREVIEW_GRAPH_NODES 1
#include <tbb/flow_graph.h>

#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBuffers.h>
#include <pvparallelview/PVZonesManager.h>
// The order here is important
#include <pvparallelview/PVZoneRenderingBCI.h>
#include <pvparallelview/PVRenderingPipelinePreprocessRouter.h>
#include <pvparallelview/PVZonesProcessor.h>

#include <functional>

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace Inendi
{
class PVSelection;
} // namespace Inendi

namespace PVParallelView
{

class PVRenderingPipelinePreprocessRouter;
class PVBCIDrawingBackend;
class PVRenderingPipeline;

class PVZonesManager;

class PVRenderingPipeline : boost::noncopyable
{
	// Structures of objects passed through the graph
	struct ZoneRenderingWithBCI {
		// Used by TBB internally
		ZoneRenderingWithBCI() {}

		ZoneRenderingWithBCI(PVZoneRenderingBCIBase_p zr_, PVBCICodeBase* codes_, size_t ncodes_)
		    : zr(std::move(zr_)), codes(codes_), ncodes(ncodes_)
		{
		}

		PVZoneRenderingBCIBase_p zr;
		PVBCICodeBase* codes;
		size_t ncodes;
	};

	typedef PVRenderingPipelinePreprocessRouter::ZoneRenderingWithColors ZoneRenderingWithColors;

	// Ports type
	typedef tbb::flow::receiver<ZoneRenderingWithColors> input_port_zrc_type;
	typedef tbb::flow::receiver<PVZoneRendering_p> input_port_cancel_type;

	/**
	 * Preprocessing before entering the pipeline.
	 */
	class Preprocessor : boost::noncopyable
	{
	  public:
		/**
		 * Type of the preprocessing function to apply for a given ZoneID
		 */
		using preprocess_func_type = std::function<void(PVZoneID)>;

	  private:
		using multinode_router = PVRenderingPipelinePreprocessRouter::multinode_router;
		using input_port_type = tbb::flow::receiver<PVZoneRendering_p>;

	  public:
		/**
		 * Build a TBB pipeline to apply preprocessing.
		 */
		Preprocessor(tbb::flow::graph& g,
		             input_port_zrc_type& node_in_job,
		             input_port_cancel_type& node_cancel_job,
		             preprocess_func_type const& f,
		             PVCore::PVHSVColor const* colors,
		             PVZonesManager const& zm);

		/**
		 * Input where we should push "token ZoneRedering"
		 */
		inline input_port_type& input_port() { return node_router; }

		/**
		 * Routing function to update state defining if preprocessing should be recomputed.
		 */
		PVRenderingPipelinePreprocessRouter& get_router() { return router; }

	  private:
		PVRenderingPipelinePreprocessRouter
		    router; //!< Routing functor on preprocess node or continue pipeline.
		tbb::flow::function_node<PVZoneRendering_p, PVZoneRendering_p>
		    node_process;             //!< Preprocessing function node.
		multinode_router node_router; //!< routing node based on router.
	};

	// Cancellation points types
	constexpr static size_t cp_continue_port = 0;
	constexpr static size_t cp_cancel_port = 1;

	// Workflow router points types
	constexpr static size_t wr_bci = 0;
	constexpr static size_t wr_scatter = 1;

	typedef tbb::flow::multifunction_node<
	    ZoneRenderingWithColors,
	    std::tuple<ZoneRenderingWithColors, PVZoneRendering_p, tbb::flow::continue_msg>>
	    cp_postlimiter_type;
	typedef tbb::flow::multifunction_node<ZoneRenderingWithBCI,
	                                      std::tuple<ZoneRenderingWithBCI, ZoneRenderingWithBCI>>
	    cp_postcomputebci_type;
	typedef tbb::flow::multifunction_node<
	    ZoneRenderingWithColors,
	    std::tuple<ZoneRenderingWithColors, ZoneRenderingWithColors>>
	    workflow_router_type;

  public:
	explicit PVRenderingPipeline(PVBCIDrawingBackend& bci_backend);
	~PVRenderingPipeline();

  public:
	PVZonesProcessor declare_processor(Preprocessor::preprocess_func_type const& f,
	                                   PVCore::PVHSVColor const* colors,
	                                   PVZonesManager const& zm);

	void cancel_all();
	void wait_for_all();

  private:
	inline tbb::flow::graph& tbb_graph() { return _g; }
	inline tbb::flow::graph const& tbb_graph() const { return _g; }

  private:
	// FIXME : Unique ptr is use as move constructor is disabled
	std::vector<std::unique_ptr<Preprocessor>> _preprocessors;
	PVBCIBuffers<BCI_BUFFERS_COUNT> _bci_buffers;

  private:
	tbb::flow::graph _g;
	tbb::flow::function_node<ZoneRenderingWithColors, ZoneRenderingWithBCI>* _node_compute_bci;
	tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>* _node_draw_bci;
	tbb::flow::function_node<ZoneRenderingWithBCI, PVZoneRendering_p>* _node_cleanup_bci;
	tbb::flow::function_node<PVZoneRendering_p>* _node_finish;
	tbb::flow::function_node<ZoneRenderingWithColors, PVZoneRendering_p>* _node_compute_scatter;
	workflow_router_type* _workflow_router;

	// Cancellation points
	cp_postlimiter_type* _cp_postlimiter;
	cp_postcomputebci_type* _cp_postcomputebci;

	tbb::flow::limiter_node<ZoneRenderingWithColors> _node_limiter;
	tbb::flow::buffer_node<ZoneRenderingWithColors> _node_buffer;
};
} // namespace PVParallelView

#endif
