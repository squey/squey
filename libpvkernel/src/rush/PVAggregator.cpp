/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>
#include <pvkernel/core/PVConfig.h>

#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <QSettings>

PVRush::PVAggregator::PVAggregator()
    : _cur_input(_inputs.begin())
    , _nstart(0)
    , _nend(PVCore::PVConfig::get()
                .config()
                .value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT)
                .toInt())
    , _begin_of_input(true)
    , _skip_lines_count(0)
    , _nread_elements(0)
    , _cur_src_index(0)
{
}

void PVRush::PVAggregator::release_inputs()
{
	for (PVRush::PVRawSourceBase_p raw_source : _inputs) {
		raw_source->release_input();
	}
}

void PVRush::PVAggregator::process_indexes(chunk_index nstart,
                                           chunk_index nend,
                                           chunk_index expected_nelts)
{
	_nstart = nstart;
	_nend = nend;

	assert(not _inputs.empty() && "Aggregator have at least one source with 0 as offset");

	// Set our aggregator accordingly
	_cur_input = _inputs.begin();
	_cur_src_index = 0;
	_begin_of_input = true;
	_nread_elements = 0;

	// Reset all inputs postion
	for (auto& src : _inputs) {
		src->seek_begin();
	}

	(*_cur_input)->prepare_for_nelts(expected_nelts);
}

PVCore::PVChunk* PVRush::PVAggregator::read_until_start_index()
{
	PVCore::PVChunk* ret = nullptr;
	bool first = true;
	while (_nstart >= _nread_elements) {
		if (not first)
			_begin_of_input = false;
		if (ret != nullptr)
			ret->free();
		if ((ret = next_chunk()) == nullptr)
			return nullptr;
		first = false;
	}
	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::next_chunk()
{
	// Get chunk from current input
	PVCore::PVChunk* ret = (*_cur_input)->operator()();

	if (ret == nullptr) {
		// No more chunks available from current input. Go on !
		_cur_input++;
		if (_cur_input == _inputs.end()) {
			// No more data available. Return nullptr
			return nullptr;
		}
		_cur_src_index = _nread_elements;
		(*_cur_input)->prepare_for_nelts(_nend - _nread_elements);
		ret = (*_cur_input)->operator()();
		_begin_of_input = true;
	}
	if (ret == nullptr) {
		throw PVRush::PVInputException((*_cur_input)->human_name().toStdString() + " is empty");
	}
	if (_begin_of_input and ret->c_elements().size() < _skip_lines_count) {
		throw PVRush::PVInputException((*_cur_input)->human_name().toStdString() +
		                               " doesn't have header");
	}

	_nread_elements += ret->c_elements().size();
	ret->_agg_index = _cur_src_index + ret->_index;

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()()
{
	if (_nread_elements >= _nend) {
		// This is the end of this job
		return nullptr;
	}

	PVCore::PVChunk* ret;
	if (not _nread_elements) {
		// We have to read until _nstart indexes to skip not required first
		// line at the beginning fo the file.
		ret = read_until_start_index();
		if (ret == nullptr) {
			PVLOG_DEBUG("Aggregator: end of inputs\n");
			return nullptr;
		}
		// First part of the chunk is useless.
		chunk_index nelts_remove = ret->c_elements().size() - (_nread_elements - _nstart);
		if (ret->_index < _skip_lines_count) {
			nelts_remove = std::max(_skip_lines_count, nelts_remove);
		}

		PVCore::list_elts& elts = ret->elements();
		PVCore::list_elts::iterator it_elt = elts.begin();
		for (chunk_index i = 0; i < nelts_remove; i++) {
			PVCore::PVElement::free(*it_elt);
			PVCore::list_elts::iterator it_er = it_elt;
			it_elt++;
			elts.erase(it_er);
		}
		_nend -= _nstart;
		_nread_elements = elts.size();
		ret->_agg_index = 0;
		_cur_src_index = -ret->_index - nelts_remove;
		_begin_of_input = false;
	} else {
		ret = next_chunk();

		if (ret == nullptr) {
			PVLOG_DEBUG("Aggregator: end of inputs\n");
			return nullptr;
		}

		if (_begin_of_input) {
			PVCore::list_elts& elts = ret->elements();
			for (size_t i = 0; i < _skip_lines_count; ++i) {
				PVCore::list_elts::iterator it = elts.begin();
				PVCore::PVElement::free(*it);
				elts.erase(it);
			}
			_begin_of_input = false;
			_nread_elements -= _skip_lines_count;
			_nend -= _skip_lines_count;
		}
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
	// Define elements index once elements to remove are removed.
	ret->set_elements_index();

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()(tbb::flow_control& fc)
{
	PVCore::PVChunk* ret = this->operator()();
	if (ret == nullptr) {
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of no more input datas\n");
		fc.stop();
	}
	return ret;
}

void PVRush::PVAggregator::add_input(PVRush::PVRawSourceBase_p in)
{
	_inputs.push_back(in);
}

void PVRush::PVAggregator::set_sources_number_fields(PVCol ncols)
{
	for (auto& raw_source : _inputs) {
		raw_source->set_number_cols_to_reserve(ncols);
	}
}
