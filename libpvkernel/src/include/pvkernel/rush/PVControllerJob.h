/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVCONTROLLERJOB_FILE_H
#define PVCONTROLLERJOB_FILE_H

#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVChunkFilterDumpElts.h>
#include <pvkernel/filter/PVChunkFilterRemoveInvalidElts.h>
#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/rush/PVPipelineTask.h>

#include <tbb/pipeline.h>

#include <memory>
#include <future>

#include <QObject>
#include <QStringList>

namespace PVRush
{

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
class PVControllerJob : public QObject
{
	// This is defined as a QObject so that Qt objects can connect to the "job done" signal
	Q_OBJECT

  public:
	typedef std::shared_ptr<PVControllerJob> p_type;
	using invalid_elements_t = std::map<size_t, std::string>;

  public:
	/*! \brief Create a PVControllerJob object.
	 */
	PVControllerJob(chunk_index begin,
	                chunk_index end,
	                PVAggregator& agg,
	                PVFilter::PVChunkFilterByElt& filter,
	                PVOutput& out_filter,
	                size_t ntokens,
	                bool compact_nraw);
	PVControllerJob(PVControllerJob const&) = delete;
	PVControllerJob(PVControllerJob&&) = delete;
	PVControllerJob& operator=(PVControllerJob const&) = delete;
	PVControllerJob& operator=(PVControllerJob&&) = delete;

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

	size_t get_value() const { return _out_filter.get_out_size(); }

  public:
	std::map<size_t, std::string> const& get_invalid_evts() const { return _inv_elts; }

  private:
	tbb::filter_t<void, void> create_tbb_filter();
	void job_has_run(); // Called when the job has finish to run

  Q_SIGNALS:
	void job_done_signal();

  private:
	// Lists
	invalid_elements_t _inv_elts; //!< Store all invalid elements.

	// Filters
	PVFilter::PVChunkFilterDumpElts _elt_invalid_filter; //!< Filter that may dump every elements.
	bool _compact_nraw;
	PVFilter::PVChunkFilterRemoveInvalidElts
	    _elt_invalid_remove; //!< Remove invalid elements from chunk to compact the NRaw

	PVAggregator& _agg;                          //!< Aggregator use to generate chunks.
	PVFilter::PVChunkFilterByElt& _split_filter; //!< Filter to split a line in multiple elements.
	PVOutput& _out_filter;                       //!< Filter Saving chunk in the NRaw.

	// Indexes are aggregator indexes !
	chunk_index _idx_begin; //!< Line number where we start extraction.
	chunk_index _idx_end;   //!< Line number where we stop extraction (excluded)
	// Number of elements to read
	chunk_index _max_n_elts; //!< Number of line we want to extract (Handle invalide elements so it
	// is not begin - end)
	size_t _ntokens; //!< Number of tokens use in the TBB pipeline process.

	// TBB doesn't provide a way to get state of the task (wether it is over or not)
	std::future<void>
	    _executor; //!< Run the TBB Pipeline in this executor to have non blocking execution
	PVPipelineTask* _pipeline; //!< The TBB pipeline performing data import.
};

typedef PVControllerJob::p_type PVControllerJob_p;
} // namespace PVRush

#endif
