/**
 * \file PVControllerJob.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVController.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <assert.h>
#include <pvkernel/core/general.h>

#define PV_MAX_INDEX 1000000000
#define PV_MAX_NELTS PICVIZ_LINES_MAX

PVRush::PVControllerJob::PVControllerJob(job_action a, int priority) :
	_elt_valid_filter(true, _all_elts),
	_elt_invalid_filter(false, _inv_elts),
	_f_nelts(&_job_done),
	_agg_tbb(nullptr)
{
	_a = a;
	_priority = priority;
	_job_done = false;
	_agg = NULL;
	_mapping_filter = NULL;
	_seq_chunk_function_filter = NULL;
	_out_filter = NULL;
	_job_finished_run = false;
	_ctrl_parent = NULL;
	_dump_all_elts = false;
	_dump_inv_elts = false;
	_job_started = false;
}

PVRush::PVControllerJob::~PVControllerJob()
{
	PVLOG_DEBUG("PVControllerJob: job has finish to run.\n");
	{
		boost::lock_guard<boost::mutex> lock(_job_finished_mut);
		_job_finished_run = true;
	}
	_job_finished.notify_all(); 
	if (_agg_tbb) {
		delete _agg_tbb;
	}
}

void PVRush::PVControllerJob::set_params(chunk_index begin, chunk_index end, chunk_index n_elts, stop_cdtion sc, PVAggregator &agg, PVFilter::PVChunkFilter_f filter, PVFilter::PVChunkFilter& mapping_filter, PVFilter::PVChunkFilter& seq_chunk_function_filter, PVOutput& out_filter, size_t nchunks, bool dump_inv_elts, bool dump_all_elts)
{
	_idx_begin = begin;
	_idx_end = end;
	_nchunks = nchunks;
	_agg = &agg;
	_filter = filter;
	_mapping_filter = &mapping_filter;
	_seq_chunk_function_filter = &seq_chunk_function_filter;
	_out_filter = &out_filter;
	_n_elts = n_elts;
	_sc = sc;

	if (sc == sc_n_elts) {
		_max_n_elts = n_elts;
		_idx_end = PV_MAX_INDEX;
	}
	else {
		_max_n_elts = end - begin;
		_n_elts = PV_MAX_NELTS;
	}
	_dump_inv_elts = dump_inv_elts;
	_dump_all_elts = dump_all_elts;
}

tbb::filter_t<void,void> PVRush::PVControllerJob::create_tbb_filter()
{
	assert(_agg);
	assert(_filter);
	assert(_out_filter);
	assert(_mapping_filter);
	assert(_seq_chunk_function_filter);

	if (_agg_tbb) {
		delete _agg_tbb;
	}
	_agg_tbb = new PVAggregatorTBB(*_agg);
	tbb::filter_t<void,PVCore::PVChunk*> input_filter(tbb::filter::serial_in_order, *_agg_tbb);

	// The source transform filter takes care of source-specific filterings
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> source_transform_filter(tbb::filter::parallel, _source_filter.f());

	// The "job" filter
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> transform_filter(tbb::filter::parallel, _filter);

	// The "pure mapping" filter
	PVFilter::PVChunkFilter_f mf = _mapping_filter->f();
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> mapped_filter(tbb::filter::parallel, mf);

	// The "sequential chunk function" filter
	PVFilter::PVChunkFilter_f cf = _seq_chunk_function_filter->f();
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> seq_chunk_filter(tbb::filter::serial_in_order, cf);
	
	// Elements count filter
	_f_nelts.done_when(_n_elts);
	tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> count_filter(tbb::filter::serial_in_order, _f_nelts.f());

	// Final output filter
	tbb::filter_t<PVCore::PVChunk*,void> out_filter(tbb::filter::serial_in_order, _out_filter->f());

	if (_dump_inv_elts | _dump_all_elts) {
		_all_elts.clear();
		_inv_elts.clear();

		tbb::filter_t<PVCore::PVChunk*, PVCore::PVChunk*> middle_chunk_filter(source_transform_filter & transform_filter & mapped_filter & seq_chunk_filter);

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

		return input_filter & middle_chunk_filter & count_filter & out_filter;
	}
	else {
		return input_filter & source_transform_filter & transform_filter & mapped_filter & seq_chunk_filter & count_filter & out_filter;
	}
}

void PVRush::PVControllerJob::job_goingto_start(PVController& ctrl)
{
	_tc_start = tbb::tick_count::now();
	_ctrl_parent = &ctrl;
	_job_started = true;
}

void PVRush::PVControllerJob::wait_end()
{
	boost::unique_lock<boost::mutex> lock(_job_finished_mut);
	while (!_job_finished_run)
		_job_finished.wait(lock);
}

bool PVRush::PVControllerJob::cancel()
{
	if (!_ctrl_parent) {
		// Job has not been launched, so can't be canceled
		return false;
	}

	_ctrl_parent->cancel_job(shared_from_this());
	return true;
}

void PVRush::PVControllerJob::job_has_run_no_output_update()
{
	_tc_end = tbb::tick_count::now();
	PVLOG_DEBUG("PVControllerJob: job has finish to run.\n");
	{
		boost::lock_guard<boost::mutex> lock(_job_finished_mut);
		_job_finished_run = true;
	}
	_job_finished.notify_all();

	emit job_done_signal();
}

void PVRush::PVControllerJob::job_has_run()
{
	_out_filter->job_has_finished();
	job_has_run_no_output_update();
}

bool PVRush::PVControllerJob::running() const
{
	return !_job_finished_run;
}

bool PVRush::PVControllerJob::done() const
{
	return _job_done;
}

int PVRush::PVControllerJob::priority() const
{
	return _priority;
}

chunk_index PVRush::PVControllerJob::idx_begin() const
{
	return _idx_begin;
}

chunk_index PVRush::PVControllerJob::idx_end() const
{
	return _idx_end;
}

chunk_index PVRush::PVControllerJob::expected_nelts() const
{
	// If we need to stop according to a number of final elements,
	// then return that number of elements. Otherwise, just returns
	// and this will be based on the start and final index of the job.
	return (_sc == sc_n_elts) ? _n_elts : 0;
}

size_t PVRush::PVControllerJob::nchunks() const
{
	return _nchunks;
}

PVRush::PVControllerJob::job_action PVRush::PVControllerJob::action() const
{
	return _a;
}

tbb::tick_count::interval_t PVRush::PVControllerJob::duration() const
{
	return _tc_end-_tc_start;
}

chunk_index PVRush::PVControllerJob::status() const
{
	return _out_filter->get_rows_count();
}

chunk_index PVRush::PVControllerJob::rejected_elements() const
{
	return _f_nelts.n_elts_invalid();
}

chunk_index PVRush::PVControllerJob::nb_elts_max() const
{
	return _max_n_elts;
}
