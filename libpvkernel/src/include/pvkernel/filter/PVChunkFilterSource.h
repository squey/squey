/**
 * \file PVChunkFilterSource.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVCHUNKFILTERSOUCE_H
#define PVFILTER_PVCHUNKFILTERSOUCE_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>

namespace PVFilter {

class LibKernelDecl PVChunkFilterSource : public PVChunkFilter {
public:
	PVChunkFilterSource();
public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk); 

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterSource)
};

}

#endif
