//! \file PVChunkFilterSource.h
//! $Id: PVChunkFilterSource.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVCHUNKFILTERSOUCE_H
#define PVFILTER_PVCHUNKFILTERSOUCE_H

#include <pvcore/general.h>
#include <pvfilter/PVChunkFilter.h>

namespace PVFilter {

class LibFilterDecl PVChunkFilterSource : public PVChunkFilter {
public:
	PVChunkFilterSource();
public:
	virtual PVCore::PVChunk* operator()(PVCore::PVChunk* chunk); 
};

}

#endif
