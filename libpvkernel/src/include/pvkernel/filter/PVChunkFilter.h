//! \file PVChunkFilter.h
//! $Id: PVChunkFilter.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

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
	virtual PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
};

typedef PVChunkFilter::func_type PVChunkFilter_f;

}

#endif
