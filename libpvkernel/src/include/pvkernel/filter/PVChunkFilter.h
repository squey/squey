/**
 * \file PVChunkFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVCHUNKFILTER_H
#define PVFILTER_PVCHUNKFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVFilterFunction.h>

namespace PVFilter {

class LibKernelDecl PVChunkFilter : public PVFilterFunctionBase<PVCore::PVChunk*,PVCore::PVChunk*>  {
public:
	PVChunkFilter();
public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

	CLASS_FILTER_NONREG(PVChunkFilter)
};

typedef PVChunkFilter::func_type PVChunkFilter_f;

}

#endif
