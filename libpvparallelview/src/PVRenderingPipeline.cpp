#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVZoneRendering.h>

#include <iostream>

#include <mcheck.h>

PVParallelView::PVRenderingPipeline::PVRenderingPipeline(PVBCIDrawingBackend& backend):
	_bci_buffers(backend),
	_node_limiter(_g, BCI_BUFFERS_COUNT),
	_node_buffer(_g)
{
	const PVBCIDrawingBackend::Flags backend_flags = backend.flags();
	// Construct static graph, without any preprocessor
	
	
	// Graph BCI processing nodes
	_node_compute_bci = new tbb::flow::function_node<ZoneRenderingWithColors, ZoneRenderingWithBCI>(
		tbb_graph(),
		tbb::flow::unlimited,
		[&](ZoneRenderingWithColors const& zrc)
		{
			PVZoneRenderingBase_p const& zr = zrc.zr;
			PVBCICodeBase* bci_buf = _bci_buffers.get_available_buffer();
			const size_t n = zr->compute_bci(zrc.colors, bci_buf);
			return ZoneRenderingWithBCI(zr, bci_buf, n);
		});
	
	// Create draw BCI function, according to backend flags and type
	bool backend_sequential = ((backend_flags & PVBCIDrawingBackend::Serial) == PVBCIDrawingBackend::Serial);
	const bool backend_sync = backend.is_sync();
	if (backend_sync) {
		_node_draw_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>(
			tbb_graph(),
			(backend_sequential) ? tbb::flow::serial : tbb::flow::unlimited,
			[&backend](ZoneRenderingWithBCI const& zrb)
			{
				zrb.zr->render_bci(backend, zrb.codes, zrb.ncodes, std::function<void()>());
				return zrb;
			}
		);
	}
	else {
		_node_draw_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, ZoneRenderingWithBCI>(
			tbb_graph(),
			(backend_sequential) ? tbb::flow::serial : tbb::flow::unlimited,
			[&](ZoneRenderingWithBCI const& zrb)
			{
				tbb::flow::receiver<ZoneRenderingWithBCI>* const recv_port = static_cast<tbb::flow::receiver<ZoneRenderingWithBCI>*>(this->_node_cleanup_bci);
				PVZoneRenderingBase* const zr = zrb.zr.get();
				zr->render_bci(backend, zrb.codes, zrb.ncodes,
					[zrb,recv_port]
					{
						recv_port->try_put(zrb);
					});
				return zrb;
			}
		);
	}

	// Cancellation points
	_cp_postlimiter = new cp_postlimiter_type(tbb_graph(), tbb::flow::unlimited,
		[](ZoneRenderingWithColors const& zrc, cp_postlimiter_type::output_ports_type& op)
		{
			if (zrc.zr->should_cancel()) {
				std::get<cp_cancel_port>(op).try_put(zrc.zr);
				// We are post the limiter, and we directly send that job to the finish node.
				// So, we need to manually trigger the limiter decrement port.
				std::get<2>(op).try_put(tbb::flow::continue_msg());
			}
			else {
				std::get<cp_continue_port>(op).try_put(zrc);
			}
		});
	_cp_postcomputebci = new cp_postcomputebci_type(tbb_graph(), tbb::flow::unlimited,
		[](ZoneRenderingWithBCI const& zrb, cp_postcomputebci_type::output_ports_type& op)
		{
			if (zrb.zr->should_cancel()) {
				std::get<cp_cancel_port>(op).try_put(zrb);
			}
			else {
				std::get<cp_continue_port>(op).try_put(zrb);
			}
		});

	// Cleanup and finish nodes
	_node_cleanup_bci = new tbb::flow::function_node<ZoneRenderingWithBCI, PVZoneRenderingBase_p>(
			tbb_graph(),
			tbb::flow::unlimited,
			[&](ZoneRenderingWithBCI const& zrb)
			{
				this->_bci_buffers.return_buffer(zrb.codes);
				this->_node_limiter.decrement.try_put(tbb::flow::continue_msg());
				return zrb.zr;
			});

	_node_finish = new tbb::flow::function_node<PVZoneRenderingBase_p>(
			tbb_graph(),
			tbb::flow::unlimited,
			[](PVZoneRenderingBase_p zr)
			{
				zr->finished(zr);
			});

	// Connect this together
	tbb::flow::make_edge(_node_buffer, _node_limiter);
	tbb::flow::make_edge(_node_limiter, *_cp_postlimiter);
	tbb::flow::make_edge(tbb::flow::output_port<cp_continue_port>(*_cp_postlimiter), *_node_compute_bci);
	tbb::flow::make_edge(*_node_compute_bci, *_cp_postcomputebci);
	tbb::flow::make_edge(tbb::flow::output_port<cp_continue_port>(*_cp_postcomputebci), *_node_draw_bci);
	if (backend_sync) {
		tbb::flow::make_edge(*_node_draw_bci, *_node_cleanup_bci);
	}
	tbb::flow::make_edge(*_node_cleanup_bci, *_node_finish);

	tbb::flow::make_edge(tbb::flow::output_port<cp_cancel_port>(*_cp_postlimiter), *_node_finish);
	tbb::flow::make_edge(tbb::flow::output_port<2>(*_cp_postlimiter), _node_limiter.decrement);
	tbb::flow::make_edge(tbb::flow::output_port<cp_cancel_port>(*_cp_postcomputebci), *_node_cleanup_bci);
}

PVParallelView::PVRenderingPipeline::~PVRenderingPipeline()
{
	cancel_all();
	wait_for_all();

	for (Preprocessor* p: _preprocessors) {
		delete p;
	}
	for (DirectInput* di: _direct_inputs) {
		delete di;
	}

	delete _node_compute_bci;
	delete _node_draw_bci;
	delete _node_cleanup_bci;
	delete _node_finish;
}

void PVParallelView::PVRenderingPipeline::cancel_all()
{
	tbb_graph().root_task()->cancel_group_execution();
}

void PVParallelView::PVRenderingPipeline::wait_for_all()
{
	tbb_graph().wait_for_all();
}

PVParallelView::PVZonesProcessor PVParallelView::PVRenderingPipeline::declare_processor(preprocess_func_type const& f, PVCore::PVHSVColor const* colors, size_t nzones)
{
	Preprocessor* pp = new Preprocessor(tbb_graph(), _node_buffer, *_node_finish, f, colors, nzones);
	_preprocessors.push_back(pp);
	return PVZonesProcessor(pp->input_port(), &pp->router);
}

PVParallelView::PVZonesProcessor PVParallelView::PVRenderingPipeline::declare_processor(PVCore::PVHSVColor const* colors)
{
	DirectInput* di = new DirectInput(tbb_graph(), _node_buffer, *_node_finish, colors);
	_direct_inputs.push_back(di);
	return PVZonesProcessor(di->node_process);
}

// Preprocess class
PVParallelView::PVRenderingPipeline::Preprocessor::Preprocessor(tbb::flow::graph& g, input_port_zrc_type& node_in_job, input_port_cancel_type& node_cancel_job, preprocess_func_type const& f, PVCore::PVHSVColor const* colors, size_t nzones):
	router(nzones, colors),
	node_process(g, 24, [=](PVZoneRenderingBase_p zr) { f(zr->get_zone_id()); return zr; }),
	node_or(g),
	node_router(g, tbb::flow::serial, router)
{
	tbb::flow::make_edge(node_or, node_router);
	tbb::flow::make_edge(tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxPreprocess>(node_router), node_process);
	tbb::flow::make_edge(tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxContinue>(node_router), node_in_job);
	tbb::flow::make_edge(tbb::flow::output_port<PVRenderingPipelinePreprocessRouter::OutIdxCancel>(node_router), node_cancel_job);
	tbb::flow::make_edge(node_process, tbb::flow::input_port<PVRenderingPipelinePreprocessRouter::InputIdxPostProcess>(node_or));
}

// DirectInput class
PVParallelView::PVRenderingPipeline::DirectInput::DirectInput(tbb::flow::graph& g, input_port_zrc_type& node_in_job, input_port_cancel_type& node_cancel_job, PVCore::PVHSVColor const* colors_):
	node_process(g, tbb::flow::unlimited,
		[=](PVZoneRenderingBase_p zr, direct_process_type::output_ports_type& op)
		{
			if (zr->should_cancel()) {
				std::get<1>(op).try_put(zr);
			}
			else {
				std::get<0>(op).try_put(ZoneRenderingWithColors(zr, colors_));
			}
		})
{
	tbb::flow::make_edge(tbb::flow::output_port<0>(node_process), node_in_job);
	tbb::flow::make_edge(tbb::flow::output_port<1>(node_process), node_cancel_job);
}
