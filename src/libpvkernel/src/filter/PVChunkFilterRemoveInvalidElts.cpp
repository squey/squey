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

#include <pvkernel/filter/PVChunkFilterRemoveInvalidElts.h>

#include <pvkernel/core/PVTextChunk.h>     // for PVChunk, list_elts
#include <pvkernel/core/PVElement.h>       // for PVElement
#include <pvkernel/filter/PVChunkFilter.h> // for PVChunkFilter

#include <list> // for _List_iterator, list

/******************************************************************************
 *
 * PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts(bool& job_done)
    : PVChunkFilter(), _current_agg_index(0), _job_done(job_done)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterRemoveInvalidElts::operator()
 *
 *****************************************************************************/
PVCore::PVTextChunk* PVFilter::PVChunkFilterRemoveInvalidElts::
operator()(PVCore::PVTextChunk* chunk)
{
	auto& elts = chunk->elements();
	auto it = elts.begin();
	chunk->set_agg_index(_current_agg_index);
	while (it != elts.end()) {
		PVCore::PVElement* elt = *it;
		if (not elt->valid() or elt->filtered()) {
			PVCore::PVElement::free(*it);
			auto it_er = it;
			++it;
			elts.erase(it_er);
		} else {
			++it;
		}
	}

	_current_agg_index += elts.size();

	// Give the information that enough data will be stored in the NRaw
	if (_current_agg_index >= EXTRACTED_ROW_COUNT_LIMIT) {
		_job_done = true;

		// Remove "extra" elements
		PVCore::list_elts& elts = chunk->elements();
		auto it_elt = elts.begin();
		std::advance(it_elt, EXTRACTED_ROW_COUNT_LIMIT - (_current_agg_index - elts.size()));

		// And remove them all till the end
		while (it_elt != elts.end()) {
			PVCore::PVElement::free(*it_elt);
			auto it_er = it_elt;
			it_elt++;
			elts.erase(it_er);
		}
	}

	return chunk;
}
