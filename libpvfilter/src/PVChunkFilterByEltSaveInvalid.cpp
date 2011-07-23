//! \file PVChunkFilterByEltSaveInvalid.cpp
//! $Id: PVChunkFilterByEltSaveInvalid.cpp 3129 2011-06-14 09:47:24Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvfilter/PVChunkFilterByEltSaveInvalid.h>
#include <pvcore/PVChunk.h>


/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltSaveInvalid::PVChunkFilterByEltSaveInvalid
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByEltSaveInvalid::PVChunkFilterByEltSaveInvalid(PVElementFilter_f elt_filter) :
	PVChunkFilter()
{
	_elt_filter = elt_filter;
	_n_elts_invalid = 0;
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltSaveInvalid::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterByEltSaveInvalid::operator()(PVCore::PVChunk* chunk)
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
		if (elt.valid()) {
			it++;
			nelts_valid++;
		}
	}
	chunk->set_elts_stat(nelts, nelts_valid);
	return chunk;
}

