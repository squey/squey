/**
 * \file PVExtractor.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/core/debug.h>
#include <iostream>

#include <tbb/task_scheduler_init.h>

PVRush::PVExtractor::PVExtractor(unsigned int chunks) :
	_ctrl_th(_ctrl),
	_out_nraw(_nraw)
{
	if (chunks == 0) {
		// Compute a value as 5 times the number of tbb's processors
		 _chunks = tbb::task_scheduler_init::default_num_threads() * 5;
		 PVLOG_DEBUG("(PVExtractor::PVExtractor) using %d chunks\n", _chunks);
	}
	else {
		_chunks = chunks;
	}
	_saved_nraw_valid = false;
	_dump_inv_elts = false;
	_dump_all_elts = false;
	_last_start = 0;
	_last_nlines = 1;
	_force_naxes = 0;
}

PVRush::PVExtractor::~PVExtractor()
{
	force_stop_controller();
}

void PVRush::PVExtractor::start_controller()
{
	// This function need to be called if you want your jobs to be processed... !
	// TODO: should we start it with a low priority ?
	_ctrl_th.start();	
}

void PVRush::PVExtractor::gracefully_stop_controller()
{
	// Graceful stop controller, which means that it will end for all jobs to stop
	_ctrl.wait_end_and_stop();
	_ctrl_th.wait();
}

void PVRush::PVExtractor::force_stop_controller()
{
	// Force the controller to stop by cancelling the current job
	_ctrl.force_stop();
	_ctrl_th.wait();
}

void PVRush::PVExtractor::add_source(PVRush::PVRawSourceBase_p src)
{
	//TODO: check if controller is running a job
	_agg.add_input(src);
}

void PVRush::PVExtractor::set_chunk_filter(PVFilter::PVChunkFilter_f chk_flt)
{
	_chk_flt = chk_flt;
}

PVRush::PVNraw& PVRush::PVExtractor::get_nraw()
{
	return _nraw;
}

PVRush::PVFormat& PVRush::PVExtractor::get_format()
{
	return *_nraw.format;
}

const PVRush::PVFormat& PVRush::PVExtractor::get_format() const
{
	return *_nraw.format;
}

chunk_index PVRush::PVExtractor::pvrow_to_agg_index(PVRow start, bool& found)
{
	chunk_index ret = 0;
	found = false;
	PVNrawOutput::map_pvrow const& mapnrow = _out_nraw.get_pvrow_index_map();
	PVNrawOutput::map_pvrow::const_iterator it_map;
	for (it_map = mapnrow.begin(); it_map != mapnrow.end(); it_map++) {
		// If the index in the Nraw if greater or equal to the one in the current map element...
		if (start >= (*it_map).first) {
			// ...that's our guy !
			ret = (*it_map).second;
			found = true;
			break;
		}
	}
	return ret;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_pvrow(PVRow start, PVRow end, int priority, bool force_process)
{
	assert(start <= end);
	set_sources_number_fields();

	// Given two pvrows, this function will create an nraw from these two indexes. If the indexes are already in the nraw,
	// it is just shrinked, unless force_process is set to true.
	
	// Find the aggregator chunk index corresponding to start
	bool idx_found;
	chunk_index idx_start = pvrow_to_agg_index(start, idx_found);
	
	// If it can't be found...
	if (!idx_found) {
		if (start != 0) {
			// Well, heh, we don't know what to do, because we can't know where to start from !
			assert(false);
		}
		// This means we need "end+1" lines from 0. Let's go !
		return process_from_agg_nlines(0, end+1, priority);
	}

	// Check whether lines from "start" to "end" already exists
	if (end < _nraw.get_table().get_nrows()) {
		if (force_process) {
			// Ok, we got them, but we want them to be reprocessed. Let's do this !
			return process_from_agg_nlines(idx_start, end-start + 1);
		}
		// Shrink the nraw
		_nraw.resize_nrows(end-start+1);
		return PVControllerJob_p(new PVControllerJobDummy());
	}
	else {
		// Process the missing ones (or all of them if force_process is set)
		if (!force_process) {
			// TODO!
			//new_idx_start = _agg.last_elt_agg_index() + 1;
		}
		return process_from_agg_nlines(idx_start, end-start+1, priority);
	}

	return PVControllerJob_p(new PVControllerJobDummy());
}


PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_nlines(chunk_index start, chunk_index nlines, int priority)
{
	set_sources_number_fields();
	_nraw.reserve(nlines, get_number_axes());

	_out_nraw.clear_pvrow_index_map();

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::start, priority));
	job->set_params(start, 0, nlines, PVControllerJob::sc_n_elts, _agg, _chk_flt, _out_nraw, _chunks, _dump_inv_elts, _dump_all_elts);
	
	// The job is submitted to the controller and the pointer returned, so that the caller can wait for its end
	_ctrl.submit_job(job);

	_last_start = start;
	_last_nlines = nlines;

	return job;
}


PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_idxes(chunk_index start, chunk_index end, int priority)
{
	set_sources_number_fields();
	_nraw.reserve(end-start, get_number_axes());

	_out_nraw.clear_pvrow_index_map();

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::start, priority));
	job->set_params(start, end, 0, PVControllerJob::sc_idx_end, _agg, _chk_flt, _out_nraw, _chunks, _dump_inv_elts, _dump_all_elts);
	
	// The job is submitted to the controller and the pointer returned, so that the caller can wait for its end
	_ctrl.submit_job(job);

	return job;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::read_everything(int priority)
{
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::read_everything, priority));
	job->set_params(0, 0, 0, PVControllerJob::sc_idx_end, _agg, _chk_flt, _out_nraw, _chunks, false, false);

	_ctrl.submit_job(job);
	return job;
}

void PVRush::PVExtractor::dump_mapnraw()
{
	PVNrawOutput::map_pvrow const& mapnrow = _out_nraw.get_pvrow_index_map();
	PVNrawOutput::map_pvrow::const_iterator it_map;
	for (it_map = mapnrow.begin(); it_map != mapnrow.end(); it_map++) {
		PVLOG_INFO("pvrow %d goes to index %d\n", (*it_map).first, (*it_map).second);
	}
}

void PVRush::PVExtractor::dump_nraw()
{
//	for (int i = 0; i < picviz_min(10,_nraw.table.size()); i++) {
//		PVLOG_INFO("Line %d: ", i);
//		debug_qstringlist(_nraw.table[i]);
//	}

	PVLOG_INFO("Nraw:\n");
	for (int i = 0; i < picviz_min(10,_nraw.get_number_rows()); i++) {
		PVLOG_INFO("Line %d: ", i);
		for (int j = 0; j < _nraw.get_number_cols(); j++) {
			std::cerr << qPrintable(_nraw.at(i,j)) << ",";
		}
		std::cerr << std::endl;
	}
}

PVRush::PVAggregator::list_inputs const& PVRush::PVExtractor::get_inputs() const
{
	return _agg.get_inputs();
}

PVRush::PVAggregator& PVRush::PVExtractor::get_agg()
{
	return _agg;
}

void PVRush::PVExtractor::debug()
{
	PVLOG_DEBUG("PVExtractor debug\n");
	_agg.debug();
	PVLOG_DEBUG("PVExtractor nraw\n");
	dump_nraw();
	PVLOG_DEBUG("PVExtractor map nraw\n");
	dump_mapnraw();
}

void PVRush::PVExtractor::save_nraw()
{
	//_saved_nraw.format.reset(new PVRush::PVFormat(*_nraw.format));
	PVNraw::swap(_saved_nraw, _nraw);
	_saved_nraw_valid = true;
}

void PVRush::PVExtractor::restore_nraw()
{
	if (_saved_nraw_valid) {
		PVNraw::swap(_nraw, _saved_nraw);
		_saved_nraw.free_trans_nraw();
		_saved_nraw_valid = false;
	}
}

void PVRush::PVExtractor::clear_saved_nraw()
{
	if (_saved_nraw_valid) {
		_saved_nraw.free_trans_nraw();
		_saved_nraw_valid = false;
	}
}

void PVRush::PVExtractor::set_format(PVFormat const& format)
{
	PVFormat* nraw_format = new PVFormat(format);
	_nraw.format.reset(nraw_format);
}

void PVRush::PVExtractor::force_number_axes(PVCol naxes)
{
	_force_naxes = naxes;
}

PVCol PVRush::PVExtractor::get_number_axes()
{
	if (_nraw.format) {
		return _nraw.format->get_axes().size();
	}
	
	// The number of axes is unknown, the NRAW will be resized
	// when the first line is created (see PVNraw::add_row)
	return _force_naxes;
}

void PVRush::PVExtractor::set_sources_number_fields()
{
	_agg.set_sources_number_fields(get_number_axes());
}

PVCore::PVArgumentList PVRush::PVExtractor::default_args_extractor()
{
	PVCore::PVArgumentList args;
	args["inv_elts"] = false;
	return args;
}
