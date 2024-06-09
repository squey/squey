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

#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/core/PVTextChunk.h>     // for list_elts, PVChunk
#include <pvkernel/core/PVElement.h>       // for PVElement
#include <pvkernel/filter/PVChunkFilter.h> // for PVChunkFilter
#include <tbb/tick_count.h> // for tick_count, operator-, etc
#include <list>
#include <cstddef> // for size_t
#include <utility>

#include "pvkernel/filter/PVElementFilter.h"

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltCancellable::PVChunkFilterByEltCancellable
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByEltCancellable::PVChunkFilterByEltCancellable(
    std::unique_ptr<PVElementFilter> elt_filter, float timeout, bool* cancellation)
    : PVChunkFilter()
    , _elt_filter(std::move(elt_filter))
    , _timeout(timeout)
    , _cancellation(cancellation)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltCancellable::operator()
 *
 *****************************************************************************/
PVCore::PVTextChunk* PVFilter::PVChunkFilterByEltCancellable::
operator()(PVCore::PVTextChunk* chunk) const
{
	PVCore::list_elts& elts = chunk->elements();
	auto it = elts.begin();
	auto ite = elts.end();
	size_t nelts = elts.size();
	size_t nelts_valid = 0;
	tbb::tick_count start = tbb::tick_count::now();
	tbb::tick_count current;

	while (it != ite) {
		tbb::tick_count ls, le;
		ls = tbb::tick_count::now();
		PVCore::PVElement& elt = (*_elt_filter)(*(*it));
		le = tbb::tick_count::now();
		if (!elt.valid()) {
			PVCore::PVElement::free(*it);
			auto it_rem = it;
			it++;
			elts.erase(it_rem);
		} else {
			it++;
			nelts_valid++;
		}

		if (_cancellation && (*_cancellation)) {
			break;
		}

		current = tbb::tick_count::now();
		if ((current - start).seconds() > _timeout) {
			break;
		}
	}
	chunk->set_elts_stat(nelts, nelts_valid);
	return chunk;
}
