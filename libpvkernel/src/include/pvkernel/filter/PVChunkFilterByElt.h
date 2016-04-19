/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELT_H
#define PVFILTER_PVCHUNKFILTERBYELT_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

/**
 * Apply filter to split the line from One PVElement to multiple.
 */
class PVChunkFilterByElt : public PVChunkFilter {
public:
	/**
	 * Build filter from the splitting function : "elt_filter".
	 */
	PVChunkFilterByElt(PVElementFilter_f elt_filter);

	/**
	 * Apply splitting to every elements from this chunk.
	 */
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

protected:
	mutable PVElementFilter_f _elt_filter; // filter to apply for splitting.

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByElt)
};

}

#endif
