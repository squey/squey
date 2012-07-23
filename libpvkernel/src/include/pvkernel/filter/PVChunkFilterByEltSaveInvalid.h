/**
 * \file PVChunkFilterByEltSaveInvalid.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTSAVEINVALID_H
#define PVFILTER_PVCHUNKFILTERBYELTSAVEINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

class LibKernelDecl PVChunkFilterByEltSaveInvalid: public PVChunkFilter {
public:
	PVChunkFilterByEltSaveInvalid(PVElementFilter_f elt_filter);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVElementFilter_f _elt_filter;
	mutable PVRow _n_elts_invalid;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByEltSaveInvalid)
};

}

#endif
