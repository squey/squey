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

#include <pvbase/general.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/core/PVBinaryChunk.h>
#include <pvkernel/core/PVTextChunk.h>
#include <cassert>

PVRush::PVControllerJob::PVControllerJob(chunk_index begin,
                                         chunk_index end,
                                         PVAggregator& agg,
                                         PVFilter::PVChunkFilterByElt& filter,
                                         PVOutput& out_filter,
                                         size_t ntokens,
                                         bool compact_nraw)
    : _elt_invalid_filter(_inv_elts)
    , _compact_nraw(compact_nraw)
    , _elt_invalid_remove(agg.job_done())
    , _agg(agg)
    , _split_filter(filter)
    , _out_filter(out_filter)
    , _idx_begin(begin)
    , _ntokens(ntokens)
{

	_max_n_elts = end - begin;
	_idx_end = end;
}

void PVRush::PVControllerJob::run_job()
{
	_executor = std::async(std::launch::async, [&]() {
		// Configure the aggregator
		_agg.process_indexes(_idx_begin, _idx_end, _max_n_elts);

		// And create the pipeline
		tbb::task_group_context ctxt; //!< Execution context of the pipeline
		_pipeline = new (tbb::task::allocate_root(ctxt)) PVPipelineTask();
		_pipeline->set_filter(create_tbb_filter());
		_pipeline->set_tokens(_ntokens);

		// Launch the pipeline
		try {
			tbb::task::spawn_root_and_wait(*_pipeline);
		} catch (...) {
			// Concider the job done if an exception raise.
			job_has_run();
			throw;
		}

		/**
		 * according to the TBB's documentation, the pipeline is implictly
		 * freed by tbb::task::spawn_root_and_wait(...).
		 */

		job_has_run();
	});
}

tbb::filter_t<void, void> PVRush::PVControllerJob::create_tbb_filter()
{
	tbb::filter_t<void, void> pause_filter(
		tbb::filter::serial_in_order, [&](tbb::flow_control& /*fc*/) { std::lock_guard<std::mutex> lg(_pause); });

	if (_agg.chunk_type() == EChunkType::TEXT) {
		tbb::filter_t<void, PVCore::PVTextChunk*> input_filter(
		    tbb::filter::serial_in_order,
		    [this](tbb::flow_control& fc) { return static_cast<PVCore::PVTextChunk*>(_agg(fc)); });

		// The "job" filter
		tbb::filter_t<PVCore::PVTextChunk*, PVCore::PVTextChunk*> transform_filter(
		    tbb::filter::parallel,
		    [this](PVCore::PVTextChunk* chunk) { return _split_filter(chunk); });

		// The next dump filter, that dumps all the invalid events
		tbb::filter_t<PVCore::PVTextChunk*, PVCore::PVTextChunk*> dump_inv_elts_filter(
		    tbb::filter::serial_out_of_order,
		    [this](PVCore::PVTextChunk* chunk) { return _elt_invalid_filter(chunk); });

		auto filter = input_filter & transform_filter & dump_inv_elts_filter;

		if (_compact_nraw) {
			// The next dump filter, that dumps all the invalid events
			tbb::filter_t<PVCore::PVTextChunk*, PVCore::PVTextChunk*> ignore_inv_elts_filter(
			    tbb::filter::serial_in_order,
			    [this](PVCore::PVTextChunk* chunk) { return _elt_invalid_remove(chunk); });
			filter = filter & ignore_inv_elts_filter;
		}

		// Final output filter
		tbb::filter_t<PVCore::PVTextChunk*, void> out_filter(
		    tbb::filter::parallel, [this](PVCore::PVTextChunk* chunk) { _out_filter(chunk); });

		return filter & out_filter & pause_filter;
	} else { // EChunkType::BINARY
		tbb::filter_t<void, PVCore::PVChunk*> input_filter(
		    tbb::filter::serial_in_order, [this](tbb::flow_control& fc) { return _agg(fc); });

		// Final output filter
		tbb::filter_t<PVCore::PVChunk*, void> out_filter(
		    tbb::filter::parallel, [this](PVCore::PVChunk* chunk) { _out_filter(chunk); });

		return input_filter & out_filter & pause_filter;
	}
}

void PVRush::PVControllerJob::wait_end()
{
	if (_executor.valid()) {
		// If it is invalid, it is already ended.
		_executor.get();
	}
}

void PVRush::PVControllerJob::cancel()
{
	pause(false);
	if (_pipeline)
		_pipeline->cancel();
	wait_end();
}

void PVRush::PVControllerJob::pause(bool pause)
{
	if (pause) {
		_pause.lock();
	}
	else {
		_pause.unlock();
	}
}

void PVRush::PVControllerJob::job_has_run()
{
	_out_filter.job_has_finished((_compact_nraw) ? invalid_elements_t{} : _inv_elts);
	Q_EMIT job_done_signal();
}

bool PVRush::PVControllerJob::running() const
{
	if (_executor.valid()) {
		return _executor.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
	} else {
		// The executor is finish for so long that it have no state anymore.
		return false;
	}
}

chunk_index PVRush::PVControllerJob::status() const
{
	return _out_filter.get_rows_count();
}

chunk_index PVRush::PVControllerJob::rejected_elements() const
{
	return _inv_elts.size();
}

chunk_index PVRush::PVControllerJob::nb_elts_max() const
{
	return _max_n_elts;
}
