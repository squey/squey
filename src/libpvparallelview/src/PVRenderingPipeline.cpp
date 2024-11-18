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

#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering.h>
#include <pvparallelview/PVZoneRenderingBCI.h>
#include <pvparallelview/PVZoneRenderingScatter.h>

#include <iostream>

PVParallelView::PVRenderingPipeline::PVRenderingPipeline(PVBCIDrawingBackend& backend)
    : _bci_buffers(backend), _node_limiter(_g, BCI_BUFFERS_COUNT), _node_buffer(_g)
{
	const PVBCIDrawingBackend::Flags backend_flags = backend.flags();
	// Construct static graph, without any preprocessor

	// Graph BCI processing nodes
	_node_compute_bci = new tbb::flow::function_node<ZoneRenderingWithColors, ZoneRenderingWithBCI>(
	    tbb_graph(), tbb::flow::unlimited, [&](ZoneRenderingWithColors const& zrc) {
		    PVZoneRenderingBCIBase_p const& zr =
		        std::static_pointer_cast<PVZoneRenderingBCIBase>(zrc.zr);
		    PVBCICodeBase* bci_buf = _bci_buffers.get_available_buffer();
		    const size_t n = zr->compute_bci(zrc.colors, bci_buf);
		    return ZoneRenderingWithBCI(zr, bci_buf, n);
		});

	// Create draw BCI function, according to backend flags and type
	bool backend_sequential =
	    ((backend_flags & PVBCIDrawingBackend::Serial) == PVBCIDrawingBackend::Serial);
	const bool backend_sync = backend.is_sync();
	if (backend_sync) {
		_node_draw_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>(
		    tbb_graph(), (backend_sequential) ? tbb::flow::serial : tbb::flow::unlimited,
		    [&backend](ZoneRenderingWithBCI const& zrb) {
			    zrb.zr->render_bci(backend, zrb.codes, zrb.ncodes, std::function<void()>());
			    return zrb;
			});
	} else {
		_node_draw_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>(
		    tbb_graph(), (backend_sequential) ? tbb::flow::serial : tbb::flow::unlimited,
		    [&](ZoneRenderingWithBCI const& zrb) {
			    auto* const recv_port =
			        static_cast<tbb::flow::receiver<ZoneRenderingWithBCI>*>(
			            this->_node_cleanup_bci);
			    PVZoneRenderingBCIBase* const zr = zrb.zr.get();
			    zr->render_bci(backend, zrb.codes, zrb.ncodes,
			                   [zrb, recv_port] { recv_port->try_put(zrb); });
			    return zrb;
			});
	}

	// Cancellation points
	_cp_postlimiter = new cp_postlimiter_type(
	    tbb_graph(), tbb::flow::unlimited,
	    [](ZoneRenderingWithColors const& zrc, cp_postlimiter_type::output_ports_type& op) {
		    if (zrc.zr->should_cancel()) {
			    std::get<cp_cancel_port>(op).try_put(zrc.zr);
			    // We are post the limiter, and we directly send that job to the
			    // finish node.
			    // So, we need to manually trigger the limiter decrement port.
			    std::get<2>(op).try_put(tbb::flow::continue_msg());
		    } else {
			    std::get<cp_continue_port>(op).try_put(zrc);
		    }
		});
	_cp_postcomputebci = new cp_postcomputebci_type(
	    tbb_graph(), tbb::flow::unlimited,
	    [](ZoneRenderingWithBCI const& zrb, cp_postcomputebci_type::output_ports_type& op) {
		    if (zrb.zr->should_cancel()) {
			    std::get<cp_cancel_port>(op).try_put(zrb);
		    } else {
			    std::get<cp_continue_port>(op).try_put(zrb);
		    }
		});

	// Cleanup and finish nodes
	_node_cleanup_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, PVZoneRendering_p>(
	    tbb_graph(), tbb::flow::unlimited, [&](ZoneRenderingWithBCI const& zrb) {
		    this->_bci_buffers.return_buffer(zrb.codes);
		    this->_node_limiter.decrementer().try_put(tbb::flow::continue_msg());
		    return zrb.zr;
		});

	_node_finish = new tbb::flow::function_node<PVZoneRendering_p>(
	    tbb_graph(), tbb::flow::unlimited, [](PVZoneRendering_p zr) {
		    // FIXME : It does nothing
		    zr->finished(zr);
		});

	// Scatter workflow
	_node_compute_scatter =
	    new tbb::flow::function_node<ZoneRenderingWithColors, PVZoneRendering_p>(
	        tbb_graph(), tbb::flow::serial, [](ZoneRenderingWithColors const& zrc) {
		        auto* zrs = static_cast<PVZoneRenderingScatter*>(zrc.zr.get());
		        zrs->render();
		        // zrs->render(zrc.colors);
		        return zrc.zr;
		    });

	// Workflow router
	_workflow_router = new workflow_router_type(
	    tbb_graph(), tbb::flow::unlimited,
	    [](ZoneRenderingWithColors const& zrc, workflow_router_type::output_ports_type& op) {
		    PVZoneRendering* zr = zrc.zr.get();
		    if (dynamic_cast<PVZoneRenderingBCIBase*>(zr)) {
			    std::get<wr_bci>(op).try_put(zrc);
		    } else {
			    std::get<wr_scatter>(op).try_put(zrc);
		    }
		});

	// Connect this together
	tbb::flow::make_edge(tbb::flow::output_port<wr_bci>(*_workflow_router), _node_buffer);
	tbb::flow::make_edge(_node_buffer, _node_limiter);
	tbb::flow::make_edge(_node_limiter, *_cp_postlimiter);
	tbb::flow::make_edge(tbb::flow::output_port<cp_continue_port>(*_cp_postlimiter),
	                     *_node_compute_bci);
	tbb::flow::make_edge(*_node_compute_bci, *_cp_postcomputebci);
	tbb::flow::make_edge(tbb::flow::output_port<cp_continue_port>(*_cp_postcomputebci),
	                     *_node_draw_bci);
	if (backend_sync) {
		tbb::flow::make_edge(*_node_draw_bci, *_node_cleanup_bci);
	}
	tbb::flow::make_edge(*_node_cleanup_bci, *_node_finish);

	tbb::flow::make_edge(tbb::flow::output_port<wr_scatter>(*_workflow_router),
	                     *_node_compute_scatter);
	tbb::flow::make_edge(*_node_compute_scatter, *_node_finish);

	tbb::flow::make_edge(tbb::flow::output_port<cp_cancel_port>(*_cp_postlimiter), *_node_finish);
	tbb::flow::make_edge(tbb::flow::output_port<2>(*_cp_postlimiter), _node_limiter.decrementer());
	tbb::flow::make_edge(tbb::flow::output_port<cp_cancel_port>(*_cp_postcomputebci),
	                     *_node_cleanup_bci);
}

PVParallelView::PVRenderingPipeline::~PVRenderingPipeline()
{
	wait_for_all();

	delete _cp_postlimiter;
	delete _cp_postcomputebci;

	delete _node_compute_bci;
	delete _node_draw_bci;
	delete _node_cleanup_bci;
	delete _node_finish;
	delete _node_compute_scatter;
	delete _workflow_router;
}

void PVParallelView::PVRenderingPipeline::wait_for_all()
{
	tbb_graph().wait_for_all();
}

PVParallelView::PVZonesProcessor
PVParallelView::PVRenderingPipeline::declare_processor(Preprocessor::preprocess_func_type const& f,
                                                       PVCore::PVHSVColor const* colors,
                                                       PVZonesManager const& zm)
{
	_preprocessors.emplace_back(
	    new Preprocessor(tbb_graph(), *_workflow_router, *_node_finish, f, colors, zm));
	return PVZonesProcessor(_preprocessors.back()->input_port(),
	                        _preprocessors.back()->get_router());
}

// Preprocess class
PVParallelView::PVRenderingPipeline::Preprocessor::Preprocessor(
    tbb::flow::graph& g,
    input_port_zrc_type& node_in_job,
    input_port_cancel_type& node_cancel_job,
    preprocess_func_type const& f,
    PVCore::PVHSVColor const* colors,
    PVZonesManager const& zm)
    : router(zm, colors)
    , node_process(g,
                   24,
                   [=,this](PVZoneRendering_p zr) {
	                   f(zr->get_zone_id());
	                   router.preprocessing_done(zr->get_zone_id());
	                   return zr;
	               })
    , node_router(g, tbb::flow::serial, router)
{
	tbb::flow::make_edge(
	    tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxPreprocess>(node_router),
	    node_process);
	tbb::flow::make_edge(
	    tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxContinue>(node_router),
	    node_in_job);
	tbb::flow::make_edge(
	    tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxCancel>(node_router),
	    node_cancel_job);
	tbb::flow::make_edge(node_process, node_router);
}
