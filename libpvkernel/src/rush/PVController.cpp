/**
 * \file PVController.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVController.h>
#include <tbb/task.h>
#include <tbb/task_scheduler_init.h>
#include <boost/thread/locks.hpp>

PVRush::PVController::PVController() :
	_shouldStop(false), _pipeline(NULL)
{
}

void PVRush::PVController::operator()()
{
	_shouldStop = false;
	while (!_shouldStop)
	{
		{
			boost::unique_lock<boost::mutex> lock(_mut_job);
			while (_jobs.size() == 0) {
				if (_shouldStop) {
					return;
				}
				_job_present.wait(lock);
			}
		}
		PVLOG_DEBUG("(PVController) Job found in queue. Processing...\n");
			
		{
			boost::mutex::scoped_lock lock(_mut_job);
			_cur_job = _jobs.front();
			_jobs.pop_front();
		}
		switch (_cur_job->action())
		{
			case PVControllerJob::start:
			{
				// Set job "job_done" flag to false
				_cur_job->_job_done = false;
				

				// Configure the aggregator
				assert(_cur_job->_agg);
				_cur_job->_agg->process_indexes(_cur_job->idx_begin(), _cur_job->idx_end(), _cur_job->expected_nelts());
				_cur_job->_agg->set_stop_condition(&(_cur_job->_job_done));
				_cur_job->_out_filter->set_stop_condition(&(_cur_job->_job_done));

				// Debug
				_cur_job->_agg->debug();

				tbb::task_scheduler_init l_anon;
				// And create the pipeline
				_pipeline = new(tbb::task::allocate_root()) PVPipelineTask();
				_pipeline->set_filter(_cur_job->create_tbb_filter());		
				_pipeline->set_nchunks(_cur_job->nchunks());

				// Launch the pipeline
				_cur_job->job_goingto_start(*this);
				tbb::task::spawn_root_and_wait(*_pipeline);

				/**
				 * according to the TBB's documentation, the pipeline is implictly
				 * freed by tbb::task::spawn_root_and_wait(...).
				 */
				_pipeline = NULL;

				_cur_job->job_has_run();

				// the current is now useless
				_cur_job.reset();

				break;
			}

			case PVControllerJob::read_everything:
			{
				assert(_cur_job->_agg);
				_cur_job->_agg->read_all_chunks_from_beggining();
				_cur_job->job_has_run_no_output_update();
				break;
			}
			
			default:
				assert(false);
		}

		PVLOG_DEBUG("(PVController) Job finished\n");

	}
}

void PVRush::PVController::submit_job(PVControllerJob_p job)
{
	boost::mutex::scoped_lock lock(_mut_job);
	bool was_empty = (_jobs.size() == 0);
	if (job->priority() == 1) {
		if (job->action() == PVControllerJob::stop_current) {
			if (_pipeline) {
				_pipeline->cancel();
			}
		}
		else {
			_jobs.push_front(job);
		}
	}
	else {
		_jobs.push_back(job);
	}
	if (was_empty)
		_job_present.notify_one();
}

void PVRush::PVController::cancel_job(PVControllerJob_p job)
{
	boost::mutex::scoped_lock lock(_mut_job);
	// If that's the current job, cancel it
	if (_cur_job == job) {
		if (_pipeline) {
			_pipeline->cancel();
			job->wait_end();
		}
		return;
	}

	// Else remove it from the queue
	list_jobs::iterator it = find(_jobs.begin(), _jobs.end(), job);	
	if (it == _jobs.end()) {
		return;
	}
	_jobs.erase(it);
}

void PVRush::PVController::force_stop()
{
	if (_pipeline)
		_pipeline->cancel();

	_ask_for_stop();
}

void PVRush::PVController::wait_end_and_stop()
{
	_ask_for_stop();
}

void PVRush::PVController::_ask_for_stop()
{
	{
		boost::lock_guard<boost::mutex> lock(_mut_job);
		_shouldStop = true;
	}
	_job_present.notify_one();
}
