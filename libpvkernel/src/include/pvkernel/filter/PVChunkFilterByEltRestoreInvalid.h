/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTRESTOREINVALID_H
#define PVFILTER_PVCHUNKFILTERBYELTRESTOREINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

class PVChunkFilterByEltRestoreInvalid: public PVChunkFilter {
public:
	PVChunkFilterByEltRestoreInvalid(PVElementFilter_f elt_filter);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVElementFilter_f _elt_filter;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByEltRestoreInvalid)
};

}

#endif
