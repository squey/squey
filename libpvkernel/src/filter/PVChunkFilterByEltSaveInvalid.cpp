/**
 * \file PVChunkFilterByEltSaveInvalid.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVChunkFilterByEltSaveInvalid.h>
#include <pvkernel/core/PVChunk.h>


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
		PVCore::PVElement &src_elt = *(*it);
		src_elt.save_elt_buffer();
		PVCore::PVElement &elt = _elt_filter(src_elt);
		if (elt.valid()) {
			nelts_valid++;
		}
		it++;
	}
	chunk->set_elts_stat(nelts, nelts_valid);
	return chunk;
}

