#include <pvrush/PVExtractor.h>
#include <pvrush/PVControllerJob.h>

#include <pvcore/debug.h>
#include <iostream>

PVRush::PVExtractor::PVExtractor(unsigned int chunks) :
	_ctrl_th(_ctrl),
	_out_nraw(_nraw)
{
	if (chunks == 0) {
		//TODO: automatically find a good value.
		_chunks = 40;
	}
	else
		_chunks = chunks;
	_saved_nraw_valid = false;
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

void PVRush::PVExtractor::add_source(PVFilter::PVRawSourceBase_p src)
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

PVCore::chunk_index PVRush::PVExtractor::pvrow_to_agg_index(PVRow start, bool& found)
{
	PVCore::chunk_index ret = 0;
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
	// Given two pvrows, this function will create an nraw from these two indexes. If the indexes are already in the nraw,
	// it is just shrinked, unless force_process is set to true.
	
	// Find the aggregator chunk index corresponding to start
	bool idx_found;
	PVCore::chunk_index idx_start = pvrow_to_agg_index(start, idx_found);
	
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
	if (end < _nraw.table.size()) {
		if (force_process) {
			// Ok, we got them, but we want them to be reprocessed. Let's do this !
			return process_from_agg_nlines(idx_start, end-start + 1);
		}
		// Shrink the nraw
		// TODO: be more efficient
		PVNraw new_nraw;
		new_nraw.table.resize(end-start+1);
		std::copy(_nraw.table.begin() + start, _nraw.table.begin()+end+1, new_nraw.table.begin());
		_nraw.table = new_nraw.table;
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


PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_nlines(PVCore::chunk_index start, PVCore::chunk_index nlines, int priority)
{
	_nraw.reserve(nlines);

	_out_nraw.clear_pvrow_index_map();

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::start, priority));
	job->set_params(start, 0, nlines, PVControllerJob::sc_n_elts, _agg, _chk_flt, _out_nraw, _chunks);
	
	// The job is submitted to the controller and the pointer returned, so that the caller can wait for its end
	_ctrl.submit_job(job);

	return job;
}


PVRush::PVControllerJob_p PVRush::PVExtractor::process_from_agg_idxes(PVCore::chunk_index start, PVCore::chunk_index end, int priority)
{
	_nraw.reserve(end-start);

	_out_nraw.clear_pvrow_index_map();

	// PVControllerJob_p is a boost shared pointer, that will automatically take care of the deletion of this
	// object when it is not needed anymore !
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::start, priority));
	job->set_params(start, end, 0, PVControllerJob::sc_idx_end, _agg, _chk_flt, _out_nraw, _chunks);
	
	// The job is submitted to the controller and the pointer returned, so that the caller can wait for its end
	_ctrl.submit_job(job);

	return job;
}

PVRush::PVControllerJob_p PVRush::PVExtractor::read_everything(int priority)
{
	PVControllerJob_p job = PVControllerJob_p(new PVControllerJob(PVControllerJob::read_everything, priority));
	job->set_params(0, 0, 0, PVControllerJob::sc_idx_end, _agg, _chk_flt, _out_nraw, _chunks);

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

	PVLOG_INFO("Trans nraw:\n");
	for (int i = 0; i < picviz_min(10,_nraw.trans_table.size()); i++) {
		PVLOG_INFO("Line %d: ", i);
		for (int j = 0; j < picviz_min(10,_nraw.table.size()); j++) {
			std::cout << qPrintable(_nraw.trans_table[i][j]) << ",";
		}
		std::cout << std::endl;
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
	PVLOG_INFO("PVExtractor debug\n");
	_agg.debug();
	PVLOG_INFO("PVExtractor nraw\n");
	dump_nraw();
	PVLOG_INFO("PVExtractor map nraw\n");
	dump_mapnraw();
}

void PVRush::PVExtractor::save_nraw()
{
	PVNraw::move(_saved_nraw, _nraw);
	_saved_nraw_valid = true;
}

void PVRush::PVExtractor::restore_nraw()
{
	if (_saved_nraw_valid) {
		PVNraw::move(_nraw, _saved_nraw);
		_saved_nraw_valid = false;
	}
}

void PVRush::PVExtractor::clear_saved_nraw()
{
	if (_saved_nraw_valid) {
		_saved_nraw.clear();
		_saved_nraw_valid = false;
	}
}
