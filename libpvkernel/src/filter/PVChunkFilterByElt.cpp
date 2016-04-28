/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/core/PVChunk.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByElt::PVChunkFilterByElt
 *
 *****************************************************************************/
PVFilter::PVChunkFilterByElt::PVChunkFilterByElt(PVElementFilter_f elt_filter) : PVChunkFilter()
{
	_elt_filter = elt_filter;
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterByElt::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterByElt::operator()(PVCore::PVChunk* chunk)
{
	size_t nelts_valid = 0;

	for (auto& elt_ : chunk->elements()) {
		PVCore::PVElement& elt = _elt_filter(*elt_);
		if (elt.valid()) {
			nelts_valid++;
		}
	}

	chunk->set_elts_stat(chunk->elements().size(), nelts_valid);
	return chunk;
}
