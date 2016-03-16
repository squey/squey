/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <assert.h>
#include <pvkernel/core/general.h>

#define PV_MAX_INDEX 1000000000

PVRush::PVControllerJob::PVControllerJob(chunk_index begin, chunk_index end, chunk_index n_elts, stop_cdtion sc,
		PVAggregator &agg, PVFilter::PVChunkFilter_f& filter, PVOutput& out_filter, size_t ntokens,
		bool dump_inv_elts, bool dump_all_elts) :
	_dump_inv_elts(dump_inv_elts),
	_dump_all_elts(dump_all_elts),
	_elt_valid_filter(true, _all_elts),
	_elt_invalid_filter(false, _inv_elts),
	_job_done(false),
	_agg(agg),
	_split_filter(filter),
	_out_filter(out_filter),
	_idx_begin(begin),
	_ntokens(ntokens)
{

	// FIXME : Should be done at compile time using tag dispatching.
	if (sc == sc_n_elts) {
		_max_n_elts = n_elts;
		_idx_end = PV_MAX_INDEX;
	}
	else {
		_max_n_elts = end - begin;
		_idx_end = end;
	}
}

void PVRush::PVControllerJob::run_read_all_job() {
	// TODO : It doesn't work but it didn't neither before....
	_executor = std::async(std::launch::async, [&](){
				_agg.read_all_chunks_from_beggining();
				job_has_run_no_output_update();
	});
}

void PVRush::PVControllerJob::run_job() {
	_executor = std::async(std::launch::async, [&](){
			_job_done = false;

			// Configure the aggregator
			_agg.process_indexes(_idx_begin, _idx_end, _max_n_elts);
			_agg.set_stop_condition(&(_job_done));
			_out_filter.set_stop_condition(&(_job_done));

			// And create the pipeline
			_pipeline = new(tbb::task::allocate_root()) PVPipelineTask();
			_pipeline->set_filter(create_tbb_filter());		
			_pipeline->set_tokens(_ntokens);

			// Launch the pipeline
			tbb::task::spawn_root_and_wait(*_pipeline);

			/**
			 * according to the TBB's documentation, the pipeline is implictly
			 * freed by tbb::task::spawn_root_and_wait(...).
			 */

			job_has_run();

	});
}

tbb::filter_t<void,void> PVRush::PVControllerJob::create_tbb_filter()
{
	tbb::filter_t<void,PVCore::PVChunk*> input_filter(tbb::filter::serial_in_order, [this](tbb::flow_control &fc){ return _agg(fc);});

	// The source transform filter takes care of source-specific filterings
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> source_transform_filter(tbb::filter::parallel, _source_filter.f());

	// The "job" filter
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> transform_filter(tbb::filter::parallel, _split_filter);

	// Final output filter
	tbb::filter_t<PVCore::PVChunk*,void> out_filter(tbb::filter::parallel, _out_filter.f());

	if (_dump_inv_elts | _dump_all_elts) {
		_all_elts.clear();
		_inv_elts.clear();

		tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> middle_chunk_filter(source_transform_filter & transform_filter);

		if (_dump_all_elts) {
			// The first dump filter, that dumps all the elements
			tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> dump_all_elts_filter(tbb::filter::serial_in_order, _elt_valid_filter.f());
			middle_chunk_filter = dump_all_elts_filter & middle_chunk_filter;
		}

		if (_dump_inv_elts) {
			// The next dump filter, that dumps all the invalid events
			tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> dump_inv_elts_filter(tbb::filter::serial_in_order, _elt_invalid_filter.f());
			middle_chunk_filter = middle_chunk_filter & dump_inv_elts_filter;
		}

		return input_filter & middle_chunk_filter & out_filter;
	}
	else {
		return input_filter & source_transform_filter & transform_filter & out_filter;
	}
}

void PVRush::PVControllerJob::wait_end()
{
	_executor.wait();
}

void PVRush::PVControllerJob::cancel()
{
	if(_pipeline)
		_pipeline->cancel();
	wait_end();
}

void PVRush::PVControllerJob::job_has_run_no_output_update()
{
	emit job_done_signal();
}

void PVRush::PVControllerJob::job_has_run()
{
	_out_filter.job_has_finished();
	job_has_run_no_output_update();
}

bool PVRush::PVControllerJob::running() const
{
	return _executor.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
}

bool PVRush::PVControllerJob::done() const
{
	return _job_done;
}

chunk_index PVRush::PVControllerJob::status() const
{
	return _out_filter.get_rows_count();
}

chunk_index PVRush::PVControllerJob::rejected_elements() const
{
	return 0; // For now, we don't rejected element, we set them to 0.
}

chunk_index PVRush::PVControllerJob::nb_elts_max() const
{
	return _max_n_elts;
}
