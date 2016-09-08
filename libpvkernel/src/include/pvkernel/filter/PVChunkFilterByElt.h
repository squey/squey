/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELT_H
#define PVFILTER_PVCHUNKFILTERBYELT_H

#include <pvkernel/filter/PVChunkFilter.h>   // for PVChunkFilter
#include <pvkernel/filter/PVElementFilter.h> // for PVElementFilter

#include <algorithm> // for move
#include <memory>    // for unique_ptr

namespace PVCore
{
class PVChunk;
}

namespace PVFilter
{

/**
 * Apply filter to split the line from One PVElement to multiple.
 */
class PVChunkFilterByElt : public PVChunkFilter
{
  public:
	PVChunkFilterByElt(std::unique_ptr<PVElementFilter> elt_filter)
	    : _elt_filter(std::move(elt_filter))
	{
	}
	/**
	 * Apply splitting to every elements from this chunk.
	 */
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk) const;

  protected:
	std::unique_ptr<PVElementFilter> _elt_filter; // filter to apply for splitting.
};
}

#endif
