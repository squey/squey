/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERSOUCE_H
#define PVFILTER_PVCHUNKFILTERSOUCE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>

namespace PVFilter {

class PVChunkFilterSource : public PVChunkFilter {
public:
	PVChunkFilterSource();
public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk); 

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterSource)
};

}

#endif
