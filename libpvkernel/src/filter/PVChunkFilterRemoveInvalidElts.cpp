/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilterRemoveInvalidElts.h>

#include <pvkernel/core/PVChunk.h>         // for PVChunk, list_elts
#include <pvkernel/core/PVElement.h>       // for PVElement
#include <pvkernel/filter/PVChunkFilter.h> // for PVChunkFilter

#include <list> // for _List_iterator, list

/******************************************************************************
 *
 * PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterRemoveInvalidElts::PVChunkFilterRemoveInvalidElts(bool& job_done)
    : PVChunkFilter(), _current_agg_index(0), _job_done(job_done)
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
			auto it_er = it;
			++it;
			elts.erase(it_er);
		} else {
			++it;
		}
	}

	_current_agg_index += elts.size();

	// Give the information that enough data will be stored in the NRaw
	if (_current_agg_index >= EXTRACTED_ROW_COUNT_LIMIT) {
		_job_done = true;

		// Remove "extra" elements
		PVCore::list_elts& elts = chunk->elements();
		auto it_elt = elts.begin();
		std::advance(it_elt, EXTRACTED_ROW_COUNT_LIMIT - (_current_agg_index - elts.size()));

		// And remove them all till the end
		while (it_elt != elts.end()) {
			PVCore::PVElement::free(*it_elt);
			auto it_er = it_elt;
			it_elt++;
			elts.erase(it_er);
		}
	}

	return chunk;
}
