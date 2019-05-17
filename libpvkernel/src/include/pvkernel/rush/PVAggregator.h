/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSHAGGREGATOR_FILE_H
#define PVRUSHAGGREGATOR_FILE_H

#include <pvkernel/rush/PVRawSourceBase_types.h>

#include <pvbase/types.h>

#include <tbb/pipeline.h>

#include <cstddef> // for size_t
#include <list>

namespace PVCore
{
class PVChunk;
} // namespace PVCore

namespace PVRush
{

/*! \brief Aggregates different sources (defined as PVRawSourceBase objects) into one source usable
 *in a TBB pipeline.
 * Each PVRawSourceBase creates PVChunk objects which contains a local index (that defines the index
 *of its first element,
 * see PVCore::PVChunk for more informations). A global starting index is stored by PVAggregator
 *that defines the global start
 * index of each source (stored in the _src_offsets object).
 *
 * Using this information, it is possible to start and finish using global indexes (used by
 *PVExtractor for instance).
 *
 * For now, each source must implement a working PVRawSourceBase::seek_begin method.
 *
 */
class PVAggregator
{
  public:
	typedef std::list<PVRush::PVRawSourceBase_p> list_sources;

  public:
	/*! \brief Create an aggregator with no source.
	 */
	PVAggregator();

	PVAggregator(const PVAggregator& org) = delete;
	PVAggregator(PVAggregator&& org) = delete;

  public:
	/*! \brief Add a source to the aggregator.
	 *  \note The number of elements of that source is not computed when it is added.
	 *
	 * \todo Add a function to compute the number of element of one source.
	 */
	void add_input(PVRush::PVRawSourceBase_p in);

  public:
	void release_inputs(bool cancel_first = false);

  public:
	/*! \brief Read a chunk from the aggregator.
	 *  \note The PVChunk object returned is allocated by one of the aggregator's sources. It is the
	 * responsability
	 *        of the caller to free it using PVChunk::free.
	 */
	PVCore::PVChunk* operator()();

	/*! \brief TBB-compatible pipeline input interface
	 */
	PVCore::PVChunk* operator()(tbb::flow_control& fc);

  public:
	EChunkType chunk_type() const;

  public:
	/*! \brief Tell the aggregator to return chunks whose global indexes are between a given range.
	 *  \param[in] nstart Global start index
	 *  \param[in] nend Global end index
	 *  \param[in] expected_nelts Defines the amount of expected elements at the end. If 0, this
	 *will not be taken into account.
	 *
	 *  After this method has been called, the next call to operator() will return a chunk which
	 *contains the element
	 *  whose global index is nstart.
	 *  Then, the operator() function will not return a chunk which contains an element whose global
	 *index is greater
	 *  than nend.
	 *
	 *  \note The first and last chunk can contains elements that are not in the given range.
	 *  \note expected_elts is used to tell the aggregator's sources the amount of expected elements
	 *that they would
	 *  have to produce. Note that they can in the end process more than this number, as some
	 *elements may have been discarded
	 *  in the middle of the process. See also PVRawSourceBase::prepare_for_nelts.
	 *
	 *  \todo Add a mode where elements of the first and final chunk that whose global index is not
	 *in the given range
	 *        are invalidated by the aggregator.
	 */
	void process_indexes(chunk_index nstart, chunk_index nend, chunk_index expected_nelts = 0);

	void set_skip_lines_count(size_t skip_lines_count) { _skip_lines_count = skip_lines_count; }

	void set_sources_number_fields(PVCol nfields);

	bool& job_done() { return _job_done; }

  protected:
	PVCore::PVChunk* read_until_start_index();
	PVCore::PVChunk* next_chunk();

  protected:
	list_sources _inputs;
	list_sources::iterator _cur_input;
	/*! \brief Indicates the end of param[in]s. Set by operator().
	 */
	chunk_index _nstart;
	chunk_index _nend;

	bool _begin_of_input;
	chunk_index _skip_lines_count;

	/*! \brief Stores the global index of the last element of the last read chunk
	 */
	chunk_index _nread_elements;
	chunk_index _cur_src_index;

	bool _job_done;
};
} // namespace PVRush

#endif
