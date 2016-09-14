/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVEXTRACTOR_FILE_H
#define PVRUSH_PVEXTRACTOR_FILE_H

#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNrawOutput.h>

#include <pvkernel/filter/PVChunkFilter.h>

#include <pvbase/general.h>
#include <pvbase/types.h>

namespace PVRush
{
class PVSourceCreator;

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
	PVExtractor(PVRush::PVFormat& format,
	            PVRush::PVNraw& nraw,
	            std::shared_ptr<PVRush::PVSourceCreator> src_plugin,
	            PVRush::PVInputType::list_inputs const& inputs);

  public:
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

	void force_number_axes(PVCol naxes);
	void release_inputs() { _agg.release_inputs(); }

	size_t max_size() const { return _max_value; }

  private:
	void set_sources_number_fields();

  private:
	PVAggregator _agg;
	PVNraw& _nraw;
	PVFormat& _format;      //!< It is the format use for extraction.
	PVNrawOutput _out_nraw; // Linked to _nraw
	PVFilter::PVChunkFilterByElt _chk_flt;
	unsigned int _chunks;
	PVCol _force_naxes;

	size_t _max_value; //!< Total size for every input handled by this extractor (metrics depend on
	                   //! inputs)
};
} // namespace PVRush

#endif
