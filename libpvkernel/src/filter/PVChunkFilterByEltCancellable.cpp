/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/core/PVChunk.h>         // for list_elts, PVChunk
#include <pvkernel/core/PVElement.h>       // for PVElement
#include <pvkernel/filter/PVChunkFilter.h> // for PVChunkFilter

#include <tbb/tick_count.h> // for tick_count, operator-, etc

#include <algorithm> // for move
#include <list>
#include <cstddef> // for size_t

namespace PVFilter
{
class PVElementFilter;
} // namespace PVFilter

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
PVCore::PVChunk* PVFilter::PVChunkFilterByEltCancellable::operator()(PVCore::PVChunk* chunk) const
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
