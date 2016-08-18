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

namespace PVRush
{

// The famous and wanted PVExtractor !!!!
/*! \brief Extract datas from an aggregator, process them through filters and write the result to an
 *NRaw
 *
 * This class owns an aggregator and a NRaw (see PVRush::PVNraw). Given a chunk filter, it process a
 *given number
 * of lines and write them to its internal NRaw.
 */
class PVExtractor
{
  public:
	PVExtractor(PVRush::PVFormat& format, PVRush::PVNraw& nraw);

  public:
	/*! \brief Add a PVRawSourceBase to the internal aggregator
	 * This function adds a source to the internal aggregator.
	 */
	void add_source(PVRush::PVRawSourceBase_p src);

	/*! \brief Process a given number of lines from a given index
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] nlines Number of lines to extract. It is
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal
	 * job controller. It can be used by the caller to wait for the end of the job (see
	 * PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_nlines(chunk_index start);

	/*! \brief Process param[in]s between indexes "start" and "end"
	 *  \param[in] start Index to start the extraction from (an index is typically a line number).
	 *  \param[in] end Index to end the extraction at
	 *  \return A PVControllerJob object that represent the job that has been pushed to the internal
	 * job controller. It can be used by the caller to wait for the end of the job (see
	 * PVControllerJob::wait_end).
	 *  \sa PVAggregator
	 */
	PVControllerJob_p process_from_agg_idxes(chunk_index start, chunk_index end);

	/*! \brief Release inputs used for load data.
	 */
	void release_inputs() { _agg.release_inputs(); }

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

  private:
	void set_sources_number_fields();

  protected:
	PVAggregator _agg;
	PVNraw& _nraw;
	PVFormat& _format;      //!< It is the format use for extraction.
	PVNrawOutput _out_nraw; // Linked to _nraw
	PVFilter::PVChunkFilterByElt _chk_flt;
	unsigned int _chunks;
	PVCol _force_naxes;

  protected:
	chunk_index _last_start;
	chunk_index _last_nlines;
};
}

#endif
