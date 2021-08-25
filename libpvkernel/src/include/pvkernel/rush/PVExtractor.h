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
	PVExtractor(const PVRush::PVFormat& format,
	            PVRush::PVOutput& output,
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

	void release_inputs(bool cancel_first = false) { _agg.release_inputs(cancel_first); }

	size_t max_size() const { return _max_value; }

  private:
	void set_sources_number_fields();

  private:
	PVAggregator _agg;
	PVOutput& _output; // Linked to _nraw
	PVFormat _format;  //!< It is the format use for extraction.
	PVFilter::PVChunkFilterByElt _chk_flt;
	unsigned int _chunks;

	size_t _max_value; //!< Total size for every input handled by this extractor (metrics depend on
	                   //! inputs)
};
} // namespace PVRush

#endif
