//! \file PVChunkFilterByElt.cpp
//! $Id: PVChunkFilterByElt.cpp 3129 2011-06-14 09:47:24Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvfilter/PVChunkFilterByElt.h>
#include <pvcore/PVChunk.h>


/******************************************************************************
 *
 * PVFilter::PVChunkFilterByElt::PVChunkFilterByElt
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByElt::PVChunkFilterByElt(PVElementFilter_f elt_filter) :
	PVChunkFilter()
{
	_elt_filter = elt_filter;
	_n_elts_invalid = 0;
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByElt::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterByElt::operator()(PVCore::PVChunk* chunk)
{
	PVCore::list_elts& elts = chunk->elements();
	PVCore::list_elts::iterator it,ite;
	it = elts.begin();
	ite = elts.end();
	size_t nelts = elts.size();
	size_t nelts_valid = 0;
	while (it != ite)
	{
		PVCore::PVElement &elt = _elt_filter(*it);
		if (!elt.valid())
		{
			PVCore::list_elts::iterator it_rem = it;
			it++;
			elts.erase(it_rem);
		}
		else {
			it++;
			nelts_valid++;
		}
	}
	chunk->set_elts_stat(nelts, nelts_valid);
	return chunk;
}

