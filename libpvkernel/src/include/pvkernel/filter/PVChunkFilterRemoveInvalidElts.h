/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVCHUNKFILTERREMOVEINVALIDELTS_H
#define PVFILTER_PVCHUNKFILTERREMOVEINVALIDELTS_H

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>

namespace PVFilter
{

/**
 * This Filter remove invalid elements and filtered elements from elements list
 * of this chunk and update agg_index accordingly to save value in a compacted
 * way in the NRaw.
 */
class PVChunkFilterRemoveInvalidElts : public PVChunkFilter
{

  public:
	PVChunkFilterRemoveInvalidElts();

	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

  protected:
	size_t _current_agg_index;
};
}

#endif
