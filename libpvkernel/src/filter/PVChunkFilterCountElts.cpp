/**
 * \file PVChunkFilterCountElts.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVChunkFilterCountElts.h>
#include <assert.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::PVChunkFilterCountElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterCountElts::PVChunkFilterCountElts(bool* done_ptr) :
	PVChunkFilter(), _n_elts(0), _n_elts_invalid(0), _done_when(0), _done_ptr(done_ptr)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::done_when
 *
 *****************************************************************************/
void PVFilter::PVChunkFilterCountElts::done_when(chunk_index n)
{
	assert(n > 0);
	_done_when = n;
	_n_elts = 0;
	_n_elts_invalid = 0;
}


/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterCountElts::operator()(PVCore::PVChunk* chunk)
{
	// Be safe for now, but in theory, each element is removed live when invalid
#if 0
	PVCore::list_elts::const_iterator it,ite;
	PVCore::list_elts const& elts = chunk->c_elements();
	ite = elts.end();
	for (it = elts.begin(); it != ite; it++) {
		if ((*it)->valid()) {
			_n_elts++;
		}
		else {
			_n_elts_invalid++;
		}
	}
#endif
	// Chunk contains the statistics added by PVChunkFilterByElts, so we have our information here !
	size_t nelts_org, nelts_valid;
	chunk->get_elts_stat(nelts_org, nelts_valid);
	_n_elts += nelts_valid;
	_n_elts_invalid += (nelts_org-nelts_valid);
	*_done_ptr = (_n_elts >= _done_when);

	return chunk;
}
