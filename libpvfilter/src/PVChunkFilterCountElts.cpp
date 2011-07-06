//! \file PVChunkFilterCountElts.cpp
//! $Id: PVChunkFilterCountElts.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvfilter/PVChunkFilterCountElts.h>
#include <assert.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::PVChunkFilterCountElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterCountElts::PVChunkFilterCountElts(bool* done_ptr) :
	PVChunkFilter(), _n_elts(0), _done_when(0), _done_ptr(done_ptr)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::done_when
 *
 *****************************************************************************/
void PVFilter::PVChunkFilterCountElts::done_when(PVCore::chunk_index n)
{
	assert(n > 0);
	_done_when = n;
	_n_elts = 0;
}


/******************************************************************************
 *
 * PVFilter::PVChunkFilterCountElts::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterCountElts::operator()(PVCore::PVChunk* chunk)
{
	// Be safe for now, but in theory, each element is removed live when invalid
	PVCore::list_elts::const_iterator it,ite;
	PVCore::list_elts const& elts = chunk->c_elements();
	ite = elts.end();
	size_t inv_elts = 0;
	for (it = elts.begin(); it != ite; it++) {
		if ((*it).valid()) {
			_n_elts++;
		}
	}
	*_done_ptr = (_n_elts >= _done_when);

	return chunk;
}

