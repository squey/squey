#ifndef PVCONTROLLERJOB_FILE_H
#define PVCONTROLLERJOB_FILE_H

#include <pvcore/general.h>
#include <pvrush/PVAggregator.h>
#include <pvcore/PVChunk.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVChunkFilterSource.h>
#include <pvfilter/PVChunkFilterCountElts.h>
#include <pvfilter/PVChunkFilterDumpElts.h>
#include <pvrush/PVOutput.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <tbb/pipeline.h>
#include <tbb/tick_count.h>

#include <QObject>
#include <QStringList>

namespace PVRush {

class PVController;

/*! \brief Defines a job that is submitted to a PVController object.
 *
 * A job has the properties defined in set_params.
 * <ul>
 * <li>A priority (0 or 1). If the priority is equal to 1, the job will be added to the top of the job queue and will be the next job processed.</li>
 * <li>An action, defined by an enum (see job_action).</li>
 * <li>The aggregator that is used for the job.</li>
 * <li>The global start index, and the global end index or the number of lines that is wanted.</li>
 * <li>A PVChunkFilter function that has to be applied.</li>
 * <li>The output filter.</li>
 * <li>The number of living chunks in the TBB pipeline.</li>
 * </ul>
 * See set_params for more informations.
 *
 * Once a job is submitted to a controller, its end can be waited by any thread by calling wait().
 * It can also be canceled by any thread by calling cancel().
 * Once a job is finished, its duration can be obtained thanks to duration().
 *
 * Moreover, a "dummy" controller job exists (PVControllerJobDummy). Its wait method returns immediatly. It is used
 * if an invalid job has to be created/returned (used by PVExtractor for instance).
 */
class LibRushDecl PVControllerJob : public QObject, public boost::enable_shared_from_this<PVControllerJob> {
friend class PVController;

	// This is defined as a QObject so that Qt objects can connect to the "job done" signal
	Q_OBJECT

public:
	typedef enum _job_action
	{
		start,
		stop_current,
		read_everything
	} job_action;

	typedef enum _stop_cdtion
	{
		sc_n_elts,
		sc_idx_end
	} stop_cdtion;

	typedef boost::shared_ptr<PVControllerJob> p_type;

private:
	PVControllerJob();

public:
	/*! \brief Create a PVControllerJob object.
	 */
	PVControllerJob(job_action a, int priority);
	virtual ~PVControllerJob();
	void set_params(PVCore::chunk_index begin, PVCore::chunk_index end, PVCore::chunk_index n_elts, stop_cdtion sc, PVAggregator &agg, PVFilter::PVChunkFilter_f filter, PVOutput& out_filter, size_t nchunks, bool dump_elts = false);
	bool done() const;
	bool running() const;
	bool cancel();
	PVCore::chunk_index status() const;
	PVCore::chunk_index nb_elts_max() const;
	virtual void wait_end(); // wait the end of this job
	tbb::tick_count::interval_t duration() const;

public:
	QStringList& get_all_elts() { return _all_elts; }
	QStringList& get_invalids_elts() { return _inv_elts; }
	
protected:
	tbb::filter_t<void,void> create_tbb_filter();
	void job_has_run(); // Called by PVController when the job has finish to run
	void job_goingto_start(PVController& ctrl); // Called by PVController when the job is going to be launched

protected:
	int priority() const;
	PVCore::chunk_index idx_begin() const;
	PVCore::chunk_index idx_end() const;
	size_t nchunks() const;
	job_action action() const;

protected:
	// For elements dumping
	bool _dump_elts;
	
	// Filters
	PVFilter::PVChunkFilterDumpElts _elt_valid_filter;
	PVFilter::PVChunkFilterDumpElts _elt_invalid_filter;

	// Lists
	QStringList _all_elts;
	QStringList _inv_elts;

protected:
	bool _job_done;
	PVAggregator* _agg;
	PVFilter::PVChunkFilter_f _filter;
	PVOutput* _out_filter;

	// Filter that count valid elements
	PVFilter::PVChunkFilterCountElts _f_nelts;
	// Source transform filter
	PVFilter::PVChunkFilterSource _source_filter;

private:
	job_action _a;
	int _priority;
	// Indexes are aggregator indexes !
	PVCore::chunk_index _idx_begin;
	PVCore::chunk_index _idx_end;
	// Number of elements to read
	PVCore::chunk_index _n_elts;
	// Stop condition (end index reached or number of elements reached)
	stop_cdtion _sc;
	size_t _nchunks;
	boost::condition_variable _job_finished;
	boost::mutex _job_finished_mut;
	bool _job_finished_run;
	tbb::tick_count _tc_start;
	tbb::tick_count _tc_end;
	PVController* _ctrl_parent;
	PVCore::chunk_index _max_n_elts;

signals:
	void job_done_signal();
};

// This class is a helper in case a "false" job has to be returned, and it won't be waited
class LibRushDecl PVControllerJobDummy : public PVControllerJob {
public:
	typedef boost::shared_ptr<PVControllerJobDummy> p_type;

public:
	PVControllerJobDummy() :
		PVControllerJob(PVControllerJob::start, 0)
	{
	}
	~PVControllerJobDummy() {}
	virtual void wait_end() { }
};

typedef PVControllerJob::p_type PVControllerJob_p;
typedef PVControllerJobDummy::p_type PVControllerJobDummy_p;

}

#endif
