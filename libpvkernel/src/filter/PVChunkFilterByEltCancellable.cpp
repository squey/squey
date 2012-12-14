/**
 * \file PVChunkFilterByElt.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/core/PVChunk.h>

#include <tbb/tick_count.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltCancellable::PVChunkFilterByEltCancellable
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByEltCancellable::PVChunkFilterByEltCancellable(PVElementFilter_f elt_filter, float timeout, bool *cancellation) :
	PVChunkFilter(),
	_timeout(timeout),
	_cancellation(cancellation)
{
	_elt_filter = elt_filter;
	_n_elts_invalid = 0;
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltCancellable::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterByEltCancellable::operator()(PVCore::PVChunk* chunk)
{
	PVCore::list_elts& elts = chunk->elements();
	PVCore::list_elts::iterator it,ite;
	it = elts.begin();
	ite = elts.end();
	size_t nelts = elts.size();
	size_t nelts_valid = 0;
	tbb::tick_count start = tbb::tick_count::now();
	tbb::tick_count current;

	while (it != ite)
	{
		tbb::tick_count ls, le;
		ls = tbb::tick_count::now();
		PVCore::PVElement &elt = _elt_filter(*(*it));
		le = tbb::tick_count::now();
		if (!elt.valid())
		{
			PVCore::PVElement::free(*it);
			PVCore::list_elts::iterator it_rem = it;
			it++;
			elts.erase(it_rem);
		}
		else {
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
