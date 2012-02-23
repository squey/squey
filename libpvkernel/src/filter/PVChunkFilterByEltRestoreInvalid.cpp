//! \file PVChunkFilterByEltRestoreInvalid.cpp
//! $Id: PVChunkFilterByEltRestoreInvalid.cpp 3129 2011-06-14 09:47:24Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/filter/PVChunkFilterByEltRestoreInvalid.h>
#include <pvkernel/core/PVChunk.h>


/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltRestoreInvalid::PVChunkFilterByEltRestoreInvalid
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByEltRestoreInvalid::PVChunkFilterByEltRestoreInvalid(PVElementFilter_f elt_filter) :
	PVChunkFilter()
{
	_elt_filter = elt_filter;
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByEltRestoreInvalid::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterByEltRestoreInvalid::operator()(PVCore::PVChunk* chunk)
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
			elt.clear_saved_buf();
			nelts_valid++;
		}
		else {
			elt.restore_elt_with_saved_buffer();
		}
		it++;
	}
	chunk->set_elts_stat(nelts, nelts_valid);
	return chunk;
}

