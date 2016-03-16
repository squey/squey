/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCONTROLLERJOB_FILE_H
#define PVCONTROLLERJOB_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterSource.h>
#include <pvkernel/filter/PVChunkFilterDumpElts.h>
#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/rush/PVPipelineTask.h>

#include <tbb/pipeline.h>

#include <memory>
#include <future>

#include <QObject>
#include <QStringList>

namespace PVRush {

/*! \brief Defines a job to import data.
 *
 * A job has the properties defined in set_params.
 * <ul>
 * <li>The aggregator that is used for the job.</li>
 * <li>The global start index, and the global end index or the number of lines that is wanted.</li>
 * <li>A PVChunkFilter function that has to be applied.</li>
 * <li>The output filter.</li>
 * <li>The number of living chunks in the TBB pipeline.</li>
 * </ul>
 *
 * Once a job is submitted to a controller, its end can be waited by any thread by calling wait().
 * It can also be canceled by any thread by calling cancel().
 */
class PVControllerJob : public QObject, public std::enable_shared_from_this<PVControllerJob>
{
	// This is defined as a QObject so that Qt objects can connect to the "job done" signal
	Q_OBJECT

public:
	typedef enum _stop_cdtion
	{
		sc_n_elts,
		sc_idx_end
	} stop_cdtion;

	typedef std::shared_ptr<PVControllerJob> p_type;

public:
	/*! \brief Create a PVControllerJob object.
	 */
	PVControllerJob(chunk_index begin, chunk_index end, chunk_index n_elts, stop_cdtion sc,
		PVAggregator &agg, PVFilter::PVChunkFilter_f& filter, PVOutput& out_filter, size_t ntokens,
		bool dump_inv_elts);
	PVControllerJob(PVControllerJob const&) = delete;
	PVControllerJob(PVControllerJob &&) = delete;
	PVControllerJob& operator=(PVControllerJob const&) = delete;
	PVControllerJob& operator=(PVControllerJob &&) = delete;

	bool done() const;
	bool running() const;
	void cancel();

	/**
	 * Return the number of rows saved in the NRaw.
	 */
	chunk_index status() const;
	chunk_index rejected_elements() const;
	chunk_index nb_elts_max() const;
	void wait_end(); // wait the end of this job

	/**
	 * Run the job outside of the Controller.
	 */
	void run_job();
	void run_read_all_job();

public:
	QStringList const& get_invalid_evts() const { return _inv_elts; }
	
public:
	tbb::filter_t<void,void> create_tbb_filter();
	void job_has_run(); // Called when the job has finish to run
	void job_has_run_no_output_update(); // Called when the job has finish to run

signals:
	void job_done_signal();

private:
	// For elements dumping
	bool _dump_inv_elts; //!< Wether we should dump invalide elements.

	// Lists
	QStringList _inv_elts; //!< Store all elements.
	
	// Filters
	PVFilter::PVChunkFilterDumpElts _elt_invalid_filter; //!< Filter that may dump every elements.

	bool _job_done; //!< Wether the job is over or not. // FIXME : It should work but it doesn't for now
	PVAggregator& _agg; //!< Aggregator use to generate chunks.
	PVFilter::PVChunkFilter_f& _split_filter; //!< Filter to split a line in multiple elements.
	PVOutput& _out_filter; //!< Filter Saving chunk in the NRaw.

	// Source transform filter
	PVFilter::PVChunkFilterSource _source_filter; //!< Filter that should be remove as it do nothing. // FIXME : To be remove

	// Indexes are aggregator indexes !
	chunk_index _idx_begin; //!< Line number where we start extraction.
	chunk_index _idx_end; //!< Line number where we stop extraction (excluded)
	// Number of elements to read
	chunk_index _max_n_elts; //!< Number of line we want to extract (Handle invalide elements so it is not begin - end)
	size_t _ntokens; //!< Number of tokens use in the TBB pipeline process.
	
	// TBB doesn't provide a way to get state of the task (wether it is over or not)
	std::future<void> _executor; //!< Run the TBB Pipeline in this executor to have non blocking execution
	PVPipelineTask* _pipeline; //!< The TBB pipeline performing data import.
};

typedef PVControllerJob::p_type PVControllerJob_p;

}

#endif
