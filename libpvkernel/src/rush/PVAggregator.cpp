/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>
#include <pvkernel/core/PVConfig.h>

#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVRawSourceBase.h>

PVRush::PVAggregator::PVAggregator()
    : _cur_input(_inputs.begin())
    , _eoi(false)
    , _nstart(0)
    , _begin_of_input(true)
    , _skip_lines_count(0)
    , _nlast(0)
    , _nend(PVCore::PVConfig::get()
                .config()
                .value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT)
                .toInt())
    , _cur_src_index(0)
    , _stop_cond(&__stop_cond_false)
    , __stop_cond_false(false)
{
}

void PVRush::PVAggregator::release_inputs()
{
	for (PVRush::PVRawSourceBase_p raw_source : _inputs) {
		raw_source->release_input();
	}
}

void PVRush::PVAggregator::set_stop_condition(bool* cond)
{
	_stop_cond = cond;
}

void PVRush::PVAggregator::process_indexes(chunk_index nstart,
                                           chunk_index nend,
                                           chunk_index expected_nelts)
{
	_nstart = nstart;
	_nend = nend;
	_eoi = false;

	// Find out the source that contains nstart

	chunk_index src_global_index = 0;
	list_inputs::iterator it_src = agg_index_to_source_iterator(nstart, &src_global_index);
	if (it_src == _inputs.end()) {
		// Unknown index, start from the beggining !
		_nlast = 0;
		_cur_src_index = 0;
		_cur_input = _inputs.begin();
		_begin_of_input = true;

		list_inputs::iterator it;
		for (it = _inputs.begin(); it != _inputs.end(); it++) {
			// Reset all inputs position pointer
			(*it)->seek_begin();
		}

		(*_cur_input)->prepare_for_nelts(expected_nelts);
		return;
	}

	// We found our source, now let's find the source offset of our index
	assert(nstart >= src_global_index);
	chunk_index src_index = nstart - src_global_index;
	chunk_index src_found_index;
	input_offset src_offset = (*it_src)->get_input_offset_from_index(src_index, src_found_index);

	// Set our aggregator accordingly
	_cur_input = it_src;
	if ((*_cur_input)->seek(src_offset)) {
		_cur_src_index = src_global_index;
		_begin_of_input = false;
		_nlast = src_global_index + src_found_index;
	} else {
		(*_cur_input)->seek_begin();
		_cur_src_index = 0;
		_begin_of_input = true;
		_nlast = src_global_index;
	}

	// Reset all inputs postion after where we are
	it_src++;
	for (; it_src != _inputs.end(); it_src++) {
		(*it_src)->seek_begin();
	}

	(*_cur_input)->prepare_for_nelts(expected_nelts);
}

PVCore::PVChunk* PVRush::PVAggregator::read_until_index(chunk_index idx) const
{
	PVCore::PVChunk* ret = NULL;
	while (_nlast < idx) {
		if (ret != NULL)
			ret->free();
		if ((ret = next_chunk()) == NULL)
			return NULL;
	}
	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::next_chunk() const
{

	// Get chunk from current input
	PVCore::PVChunk* ret = (*_cur_input)->operator()();

	while (ret == NULL) {
		// No more chunks available from current input. Go on !
		_cur_input++;
		if (_cur_input == _inputs.end()) {
			// No more data available. Return NULL
			return NULL;
		}
		_cur_src_index = _nlast;
		_src_offsets[_cur_src_index] = _cur_input;
		(*_cur_input)->prepare_for_nelts(_nend - _nlast);
		ret = (*_cur_input)->operator()();
		_begin_of_input = true;
	}

	ret->_agg_index = _cur_src_index + ret->_index;
	if (_begin_of_input) {
		PVCore::list_elts& elts = ret->elements();
		for (size_t i = 0; i < _skip_lines_count; ++i) {
			PVCore::list_elts::iterator it = elts.begin();
			if (it != elts.end()) {
				PVCore::PVElement::free(*it);
				elts.erase(it);
			}
		}
		_begin_of_input = false;
	}
	_nlast += ret->c_elements().size();

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()() const
{
	if (*_stop_cond) {
		PVLOG_DEBUG("(PVAggregator) aggregator stop because of stop condition\n");
		return NULL;
	}

	if (_nlast >= _nend) {
		// This is the end of this job
		return NULL;
	}

	PVCore::PVChunk* ret;
	if (_nlast < _nstart) {
		// We have to read until _nstart indexes
		ret = read_until_index(_nstart);
		if (ret != NULL && ret->_agg_index < _nstart) {

			assert(ret->_agg_index + ret->c_elements().size() >= _nstart);

			chunk_index nelts_remove = _nstart - ret->_agg_index;
			PVCore::list_elts& elts = ret->elements();
			PVCore::list_elts::iterator it_elt = elts.begin();
			for (chunk_index i = 0; i < nelts_remove; i++) {
				PVCore::PVElement::free(*it_elt);
				PVCore::list_elts::iterator it_er = it_elt;
				it_elt++;
				elts.erase(it_er);
			}
			ret->_agg_index += nelts_remove;
			ret->_index += nelts_remove;
		}
	} else {
		ret = next_chunk();
	}

	if (ret == NULL) {
		_eoi = true;
		PVLOG_DEBUG("Aggregator: end of inputs\n");
		return NULL;
	}

	chunk_index nelts = ret->c_elements().size();
	if (ret->_agg_index + nelts > _nend) {
		// We need to shrink that last chunk
		// As we use std::list for elements, this will not be
		// really efficient.
		// TODO: profile this.
		chunk_index nstart_rem = _nend + 1 - ret->_agg_index;
		PVCore::list_elts& elts = ret->elements();
		PVCore::list_elts::iterator it_elt = elts.begin();
		// Go to the nstart_rem ith element
		for (chunk_index i = 0; i < nstart_rem; i++) {
			it_elt++;
		}

		// And remove them all till the end
		while (it_elt != elts.end()) {
			PVCore::PVElement::free(*it_elt);
			PVCore::list_elts::iterator it_er = it_elt;
			it_elt++;
			elts.erase(it_er);
		}
	}

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()(tbb::flow_control& fc) const
{
	PVCore::PVChunk* ret = this->operator()();
	if (ret == NULL) {
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of no more input datas\n");
		fc.stop();
	}
	return ret;
}

void PVRush::PVAggregator::add_input(PVRush::PVRawSourceBase_p in)
{
	_inputs.push_back(in);
	if (_inputs.size() == 1) {
		_src_offsets[0] = _inputs.begin();
	}
}

PVRush::PVAggregator::list_inputs::iterator
PVRush::PVAggregator::agg_index_to_source_iterator(chunk_index idx, chunk_index* global_index)
{
	map_source_offsets::reverse_iterator it;
	for (it = _src_offsets.rbegin(); it != _src_offsets.rend(); it++) {
		if (idx >= (*it).first) {
			if (global_index)
				*global_index = (*it).first;
			return it->second;
		}
	}
	return _inputs.end();
}

void PVRush::PVAggregator::set_sources_number_fields(PVCol ncols)
{
	for (auto& raw_source : _inputs) {
		raw_source->set_number_cols_to_reserve(ncols);
	}
}
