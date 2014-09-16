/**
 * \file PVControllerJob.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCONTROLLERJOB_FILE_H
#define PVCONTROLLERJOB_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterSource.h>
#include <pvkernel/filter/PVChunkFilterCountElts.h>
#include <pvkernel/filter/PVChunkFilterDumpElts.h>
#include <pvkernel/rush/PVOutput.h>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

#include <tbb/pipeline.h>
#include <tbb/tick_count.h>

#include <memory>

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
class LibKernelDecl PVControllerJob : public QObject, public std::enable_shared_from_this<PVControllerJob>, boost::noncopyable
{
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

	typedef std::shared_ptr<PVControllerJob> p_type;

private:
	PVControllerJob();

public:
	/*! \brief Create a PVControllerJob object.
	 */
	PVControllerJob(job_action a, int priority);
	virtual ~PVControllerJob();
	void set_params(chunk_index begin, chunk_index end, chunk_index n_elts, stop_cdtion sc, PVAggregator &agg, PVFilter::PVChunkFilter_f filter, PVFilter::PVChunkFilter& mapping_filter, PVFilter::PVChunkFilter& seq_chunk_function_filter, PVOutput& out_filter, size_t nchunks, bool dump_inv_elts = false, bool dump_all_elts = false);
	bool done() const;
	bool running() const;
	bool cancel();
	bool started() const { return _job_started; }
	chunk_index status() const;
	chunk_index rejected_elements() const;
	chunk_index nb_elts_max() const;
	virtual void wait_end(); // wait the end of this job
	tbb::tick_count::interval_t duration() const;
	size_t total_bytes_processed() const { return _agg_tbb->total_bytes(); }
	// In MB/s
	double mean_bw() const { return ((double)(total_bytes_processed())/(1024.0*1024.0))/duration().seconds(); }

public:
	QStringList& get_all_elts() { return _all_elts; }
	QStringList& get_invalid_evts() { return _inv_elts; }

	QStringList const& get_all_elts() const { return _all_elts; }
	QStringList const& get_invalid_evts() const { return _inv_elts; }
	
protected:
	tbb::filter_t<void,void> create_tbb_filter();
	void job_has_run(); // Called by PVController when the job has finish to run
	void job_has_run_no_output_update(); // Called by PVController when the job has finish to run
	void job_goingto_start(PVController& ctrl); // Called by PVController when the job is going to be launched

protected:
	int priority() const;
	chunk_index idx_begin() const;
	chunk_index idx_end() const;
	chunk_index expected_nelts() const;
	size_t nchunks() const;
	job_action action() const;

protected:
	// For elements dumping
	bool _dump_inv_elts;
	bool _dump_all_elts;
	
	// Filters
	PVFilter::PVChunkFilterDumpElts _elt_valid_filter;
	PVFilter::PVChunkFilterDumpElts _elt_invalid_filter;

	// Lists
	QStringList _all_elts;
	QStringList _inv_elts;

protected:
	bool _job_done;
	bool _job_started;
	PVAggregator* _agg;
	PVFilter::PVChunkFilter_f _filter;
	PVFilter::PVChunkFilter* _mapping_filter;
	PVFilter::PVChunkFilter* _seq_chunk_function_filter;
	PVOutput* _out_filter;

	// Filter that count valid elements
	PVFilter::PVChunkFilterCountElts _f_nelts;
	// Source transform filter
	PVFilter::PVChunkFilterSource _source_filter;

private:
	job_action _a;
	int _priority;
	// Indexes are aggregator indexes !
	chunk_index _idx_begin;
	chunk_index _idx_end;
	// Number of elements to read
	chunk_index _n_elts;
	// Stop condition (end index reached or number of elements reached)
	stop_cdtion _sc;
	size_t _nchunks;
	boost::condition_variable _job_finished;
	boost::mutex _job_finished_mut;
	bool _job_finished_run;
	tbb::tick_count _tc_start;
	tbb::tick_count _tc_end;
	PVController* _ctrl_parent;
	chunk_index _max_n_elts;
	
	PVAggregatorTBB* _agg_tbb;

signals:
	void job_done_signal();
};

// This class is a helper in case a "false" job has to be returned, and it won't be waited
class LibKernelDecl PVControllerJobDummy : public PVControllerJob {
public:
	typedef std::shared_ptr<PVControllerJobDummy> p_type;

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
