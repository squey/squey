#ifndef PVCONTROLLER_FILE_H
#define PVCONTROLLER_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVPipelineTask.h>
#include <pvrush/PVControllerJob.h>

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include <tbb/compat/thread>


#include <deque>

namespace PVRush {

// 
/*! \brief Job queue processing with priorities.
 * 
 * This class process jobs (represented by PVControllerJob objects) one by one with a system of priority that
 * can put a job at the beggining of the queue if its priority is one.
 *
 * \note This class should be launched in a separate thread. See PVControllerThread for instance
 * \sa PVControllerThread, PVExtractor
 */
class LibRushDecl PVController {
protected:
	typedef std::deque<PVControllerJob_p> list_jobs;

public:
	PVController();

public:
	/*! \brief Put a job in the queue
	 */
	void submit_job(PVControllerJob_p job);

	/*! \brief Cancel a job.
	 * If this job is running, it will stop its execution. Otherwise, it will remove it
	 * from the que.
	 */
	void cancel_job(PVControllerJob_p job);

public:
	/*! \brief Controller main method. This should be called in a separate thread.
	 *  \sa PVControllerThread
	 */
	void operator()();

	/*! \brief Force the controller to stop.
	 * If a job is running, this will cancel it.
	 */
	void force_stop();

	/*! \brief Wait for the end of the current job and stop the controller.
	 */
	void wait_end_and_stop();

private:
	void _ask_for_stop();

protected:
	list_jobs _jobs;
	bool _shouldStop;
	PVPipelineTask *_pipeline;
	boost::condition_variable _job_present;
	boost::mutex _mut_job;
	PVControllerJob_p _cur_job;
};

}

#endif
