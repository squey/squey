/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTSAVEINVALID_H
#define PVFILTER_PVCHUNKFILTERBYELTSAVEINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

class PVChunkFilterByEltSaveInvalid: public PVChunkFilterByElt {
public:
	PVChunkFilterByEltSaveInvalid(PVElementFilter_f elt_filter);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVRow _n_elts_invalid;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByEltSaveInvalid)
};

}

#endif
