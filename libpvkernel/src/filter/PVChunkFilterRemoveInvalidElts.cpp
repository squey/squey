/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilterRemoveInvalidElts.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts()
    : PVChunkFilter(), _current_agg_index(0)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterRemoveInvalidElts::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterRemoveInvalidElts::operator()(PVCore::PVChunk* chunk)
{
	auto& elts = chunk->elements();
	auto it = elts.begin();
	chunk->set_agg_index(_current_agg_index);
	while (it != elts.end()) {
		PVCore::PVElement* elt = *it;
		if (not elt->valid() or elt->filtered()) {
			PVCore::PVElement::free(*it);
			PVCore::list_elts::iterator it_er = it;
			++it;
			elts.erase(it_er);
		} else {
			++it;
		}
	}

	_current_agg_index += elts.size();

	return chunk;
}
