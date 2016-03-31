/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVEXTRACTOR_FILE_H
#define PVRUSH_PVEXTRACTOR_FILE_H

#include <pvbase/general.h>

#include <pvbase/types.h>
#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>

namespace PVRush {

// The famous and wanted PVExtractor !!!!
/*! \brief Extract datas from an aggregator, process them through filters and write the result to an NRaw
 *
 * This class owns an aggregator and a NRaw (see PVRush::PVNraw). Given a chunk filter, it process a given number
 * of lines and write them to its internal NRaw.
 */
class PVExtractor {
public:
	PVExtractor();
public:

	/*! \brief Add a PVRawSourceBase to the internal aggregator
	 * This function adds a source to the internal aggregator.
	 */
	void add_source(PVRush::PVRawSourceBase_p src);

	/*! \brief Set the chunk filter used during the extraction
	 * \param[in] chk_flt A boost::function object of the corresponding PVChunkFilter (can be easily obtained via the PVChunkFilter::f() method)
	 *
	 * \note It is the responsability of the caller to have the pointer to the original PVChunkFilter object valid.
	 */
	void set_chunk_filter(PVFilter::PVChunkFilterByElt* chk_flt);

	/*! \brief Process a given number of lines from a given index
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] nlines Number of lines to extract. It is 
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_nlines(chunk_index start, chunk_index nlines);

	/*! \brief Process param[in]s between indexes "start" and "end"
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] end Index to end the extraction at
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_idxes(chunk_index start, chunk_index end);

	/*! \brief 
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal job controller. It can be used by the caller to wait for the end of the job (see PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p read_everything();

	/*! \brief Get the list of sources of the internal aggregator
	 */
	PVAggregator::list_inputs const& get_inputs() const;

	/*! \brief Get a reference to the internal aggregator
	 */
	PVAggregator& get_agg() { return _agg; }
	PVAggregator const& get_agg() const { return _agg; }

	/*! \brief Get a reference to the internal NRaw
	 */
	inline PVNraw& get_nraw() { assert(_nraw); return *_nraw; }
	inline PVNraw const& get_nraw() const { assert(_nraw); return *_nraw; }

	/*! \brief Get a reference to the internal PVFormat of the internal NRaw
	 */
	PVFormat& get_format();
	PVFormat const& get_format() const;

	/*! \brief Set the format of the NRaw
	 */
	void set_format(PVFormat const& format);

	/*! \brief Clear the current nraw and saved nraw, and create a new empty one.
	 */
	void reset_nraw();

	void force_number_axes(PVCol naxes);

	chunk_index get_last_start() const { return _last_start; }
	chunk_index get_last_nlines() const { return _last_nlines; }
	void set_last_start(chunk_index start) { _last_start = start; }
	void set_last_nlines(chunk_index nlines) { _last_nlines = nlines; }
	inline void set_number_living_chunks(unsigned int nchunks)
	{
		if (nchunks > 0) {
			_chunks = nchunks;
		}
	}

	static PVCore::PVArgumentList default_args_extractor();

private:
	void set_sources_number_fields();
	
protected:
	PVAggregator _agg;
	PVFormat _format; //!< It is the format use for extraction.
	std::unique_ptr<PVNraw> _nraw;
	PVNrawOutput _out_nraw; // Linked to _nraw
	PVFilter::PVChunkFilterByElt* _chk_flt;
	unsigned int _chunks;
	PVCol _force_naxes; 

protected:
	chunk_index _last_start;
	chunk_index _last_nlines;
	
};

}

#endif
