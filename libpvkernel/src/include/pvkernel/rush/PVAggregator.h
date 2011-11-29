/*
 * $Id: PVAggregator.h 3206 2011-06-27 11:45:45Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVRUSHAGGREGATOR_FILE_H
#define PVRUSHAGGREGATOR_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>

#include <tbb/pipeline.h>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

#define DEFAULT_NUMBER_LINES 1000000

namespace PVRush {

/*! \brief Aggregates different sources (defined as PVRawSourceBase objects) into one source usable in a TBB pipeline.
 * Each PVRawSourceBase creates PVChunk objects which contains a local index (that defines the index of its first element,
 * see PVCore::PVChunk for more informations). A global starting index is stored by PVAggregator that defines the global start
 * index of each source (stored in the _src_offsets object).
 *
 * Using this information, it is possible to start and finish using global indexes (used by PVExtractor for instance).
 *
 * For now, each source must implement a working PVRawSourceBase::seek_begin method.
 *
 */
class LibKernelDecl PVAggregator {
	friend class PVSerializeObject;
public:
	typedef std::list<PVRush::PVRawSourceBase_p> list_inputs;
	typedef boost::shared_ptr<PVAggregator> p_type;
	typedef std::map<chunk_index, list_inputs::iterator> map_source_offsets;

public:
	/*! \brief Create an aggregator from a list of PVRawSourceBase objects stored as shared pointers.
	 */
	PVAggregator(list_inputs const& inputs);

	/*! \brief Create an aggregator with no source.
	 */
	PVAggregator();

private:
	/*! \brief Copy constructor of an aggregator.
	 * It is a private constructor as PVAggregator must be used as shared objects.
	 */
	PVAggregator(const PVAggregator& org);

public:

	/*! \brief Add a source to the aggregator.
	 *  \note The number of elements of that source is not computed when it is added. See read_all_chunks_from_beggining
	 *        for this purpose.
	 *
	 * \todo Add a function to compute the number of element of one source.
	 */
	void add_input(PVRush::PVRawSourceBase_p in);

public:
	/*! \brief Read a chunk from the aggregator.
	 *  \note The PVChunk object returned is allocated by one of the aggregator's sources. It is the responsability
	 *        of the caller to free it using PVChunk::free.
	 */
	PVCore::PVChunk* operator()() const;

	/*! \brief TBB-compatible pipeline input interface
	 */
	PVCore::PVChunk* operator()(tbb::flow_control &fc) const;

public:
	/*! \brief Tell the aggregator to return chunks whose global indexes are between a given range.
	 *  \param[in] nstart Global start index
	 *  \param[in] nend Global end index
	 *  \param[in] expected_nelts Defines the amount of expected elements at the end. If 0, this will not be taken into account.
	 *
	 *  After this method has been called, the next call to operator() will return a chunk which contains the element
	 *  whose global index is nstart.
	 *  Then, the operator() function will not return a chunk which contains an element whose global index is greater
	 *  than nend.
	 *
	 *  \note The first and last chunk can contains elements that are not in the given range.
	 *  \note expected_elts is used to tell the aggregator's sources the amount of expected elements that they would
	 *  have to produce. Note that they can in the end process more than this number, as some elements may have been discarded
	 *  in the middle of the process. See also PVRawSourceBase::prepare_for_nelts.
	 *
	 *  \todo Add a mode where elements of the first and final chunk that whose global index is not in the given range
	 *        are invalidated by the aggregator.
	 */
	void process_indexes(chunk_index nstart, chunk_index nend, chunk_index expected_nelts = 0);

	/*! \brief Tell the aggregator to return chunk starting from a given param[in] and a given local index range.
	 * \param[in] input_start Source to start from
	 * \param[in] nstart Local start index
	 * \param[in] nend Local end index
	 *
	 * This function retrieve the global index of param[in]_start and call process_indexes with the global version of nstart
	 * and nend.
	 *
	 * \sa process_indexes
	 * \note Because process_indexes is used, if nstart is greater than then umber of elements of param[in]_start, then the following
	 *       source will be used.
	 */
	void process_from_source(list_inputs::iterator input_start, chunk_index nstart, chunk_index nend);

	/*! \brief Returns true if the end of param[in]s has been reached. It is set by operator(). Returns false otherwise.
	 */
	bool eoi() const;

	/*! \brief Set a pointer to a stop condition.
	 *  \param[in] cond Pointer to a bool variable to represent the stop condition.
	 *
	 * operator() will return NULL if *cond is TRUE.
	 *
	 * \sa PVRush::PVController::operator()
	 */
	void set_stop_condition(bool* cond);

	/*! \brief Returns the global index of the last element of the last read chunk.
	 */
	chunk_index last_elt_agg_index();

	/*! \brief Reads all the aggregator's sources. 
	 * This will set the number of element of each sources (see PVRawSource::operator()) and updates
	 * _src_offsets.
	 *
	 * \todo Find an elegant way to be efficient.
	 */
	void read_all_chunks_from_beggining();

	/*! \brief Returns a list of the aggregator param[in]s.
	 *  \return a const-reference tothe std::vector object that stores the shared pointers to the sources.
	 */
	list_inputs const& get_inputs() const;

	/*! \brief Find the source that contains the given global index.
	 *  \param[in] idx Global index to search
	 *  \param[out] index If not NULL, the global index of the first element of the found input source.
	 *  \return A shared pointer to the source that contains the given global index if found, or an invalid shared
	 *          pointer otherwise.
	 */
	PVRush::PVRawSourceBase_p agg_index_to_source(chunk_index idx, chunk_index* global_index);

	/*! \brief Set strict mode on or off.
	 *  \param[in] sm Set to true to enable strict mode.
	 *
	 *  "Strict mode" tells the aggregator not to have elments whose global index
	 *  is above what has been asked.
	 */
	void set_strict_mode(bool sm) { _strict_mode = sm; }

	void debug();

public:
	/*! \brief Helper static function to create a PVAggregator object from a unique source.
	 */
	static p_type from_unique_source(PVRush::PVRawSourceBase_p source);

protected:
	PVCore::PVChunk* read_until_index(chunk_index idx) const;
	bool read_until_source(list_inputs::iterator input_start);
	PVCore::PVChunk* next_chunk() const;
	list_inputs::iterator agg_index_to_source_iterator(chunk_index idx, chunk_index* global_index);


	void init();

protected:
	list_inputs _inputs;
	mutable list_inputs::iterator _cur_input;
	/*! \brief Indicates the end of param[in]s. Set by operator().
	 */
	mutable bool _eoi;
	mutable chunk_index _nstart;

	/*! \brief Stores the global index of the last element of the last read chunk
	 */
	mutable chunk_index _nlast;
	mutable chunk_index _nend;
	mutable chunk_index _cur_src_index;
	bool *_stop_cond;
	bool __stop_cond_false;
	mutable chunk_index _last_elt_agg_index;

	/*! \brief Map global start indexes to source.
	 * The key of this std::map object represent the global start index of the associated source.
	 * For instance, if an aggregator contains 2 text files, this map object will contain the following information:
	 * <ul>
	 * <li>[Global index 0] -> first source</li>
	 * <li>[Number of elements of first source] -> second source</li>
	 * </ul>
	 *
	 * These informations are not computed each time a source is added, and are stored as soon as they are known
	 * by the aggregator. See read_all_chunks_from_beggining for this purpose.
	 *
	 * \note Global indexes start at 0.
	 */
	mutable map_source_offsets _src_offsets;

	/*! \brief Strict mode
	 * An aggregator in strict mode will strictely respect the start and end indexes that are
	 * given to process_indexes. This is used for instance in the format builder.
	 */
	bool _strict_mode;
};

/*! \brief Helper class to use a reference to an aggregator as a TBB filter.
 *  \sa PVAggregator copy constructor.
 */
class LibKernelDecl PVAggregatorTBB {
public:
	PVAggregatorTBB(PVAggregator &ref) :
		_ref(ref)
	{
	}

	inline PVCore::PVChunk* operator()(tbb::flow_control &fc) const
	{
		return _ref(fc);
	}

protected:
	PVAggregator &_ref;
};

typedef PVAggregator::p_type PVAggregator_p;

}


#endif
