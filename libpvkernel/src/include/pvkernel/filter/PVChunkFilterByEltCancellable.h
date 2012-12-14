/**
 * \file PVChunkFilterByEltCancellable.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H
#define PVFILTER_PVCHUNKFILTERBYELTCANCELLABLE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

class LibKernelDecl PVChunkFilterByEltCancellable : public PVChunkFilter {
public:
	PVChunkFilterByEltCancellable(PVElementFilter_f elt_filter, float timeout,
	                              bool *cancellation = nullptr);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVElementFilter_f _elt_filter;
	mutable PVRow _n_elts_invalid;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByEltCancellable)

private:
	float  _timeout;
	bool  *_cancellation;
};

}

#endif
