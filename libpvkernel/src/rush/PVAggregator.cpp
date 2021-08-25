//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVAggregator.h>          // for PVAggregator, etc
#include <pvkernel/rush/PVInput.h>               // for PVInputException
#include <pvkernel/rush/PVRawSourceBase.h>       // for PVRawSourceBase
#include <pvkernel/rush/PVRawSourceBase_types.h> // for PVRawSourceBase_p

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVConfig.h>  // for PVConfig
#include <pvkernel/core/PVElement.h> // for PVElement
#include <pvkernel/core/PVLogger.h>  // for PVLOG_DEBUG

#include <pvbase/general.h>
#include <pvbase/types.h> // for chunk_index, PVCol

#include <tbb/pipeline.h> // for flow_control

#include <cstddef>   // for size_t
#include <algorithm> // for max
#include <cassert>   // for assert
#include <list>      // for _List_iterator, etc
#include <memory>    // for allocator, __shared_ptr, etc
#include <string>    // for operator+, basic_string, etc

#include <QSettings>

PVRush::PVAggregator::PVAggregator()
    : _cur_input(_inputs.begin())
    , _nstart(0)
    , _nend(0)
    , _begin_of_input(true)
    , _skip_lines_count(0)
    , _nread_elements(0)
    , _cur_src_index(0)
    , _job_done(false)
{
}

void PVRush::PVAggregator::release_inputs(bool cancel_first /* = false */)
{
	for (list_sources::iterator it = _cur_input; it != _inputs.end(); it++) {
		(*it)->release_input(cancel_first);
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
		// Release input on-the-fly to reduce resources consumption
		(*_cur_input)->release_input(false);
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
	if (_begin_of_input and ret->rows_count() < _skip_lines_count) {
		throw PVRush::PVInputException((*_cur_input)->human_name().toStdString() +
		                               " doesn't have header");
	}

	_nread_elements += ret->rows_count();
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
		chunk_index nelts_remove = ret->rows_count() - (_nread_elements - _nstart);
		if (ret->_index < _skip_lines_count) {
			nelts_remove = std::max(_skip_lines_count, nelts_remove);
		}
		ret->remove_nelts_front(nelts_remove);
		_nend -= _nstart;
		_nread_elements = ret->rows_count();
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
			ret->remove_nelts_front(_skip_lines_count);
			_begin_of_input = false;
			_nread_elements -= _skip_lines_count;
			_nend -= _skip_lines_count;
			_cur_src_index -= _skip_lines_count;
		}
	}

	chunk_index nelts = ret->rows_count();
	if (ret->_agg_index + nelts > _nend) {
		// We need to shrink that last chunk
		// As we use std::list for elements, this will not be
		// really efficient.
		// TODO: profile this.
		chunk_index nstart_rem = _nend + 1 - ret->_agg_index;
		ret->remove_nelts_front(nelts - nstart_rem, nstart_rem);
	}
	// Define elements index once elements to remove are removed.
	ret->set_elements_index();

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()(tbb::flow_control& fc)
{
	if (_job_done) {
		fc.stop();
		return nullptr;
	}
	PVCore::PVChunk* ret = this->operator()();
	if (ret == nullptr) {
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of no more input datas\n");
		fc.stop();
	}
	return ret;
}

PVRush::EChunkType PVRush::PVAggregator::chunk_type() const
{
	return (*_cur_input)->chunk_type();
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
