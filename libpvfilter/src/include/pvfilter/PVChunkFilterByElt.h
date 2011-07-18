//! \file PVChunkFilterByElt.h
//! $Id: PVChunkFilterByElt.h 3129 2011-06-14 09:47:24Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVCHUNKFILTERBYELT_H
#define PVFILTER_PVCHUNKFILTERBYELT_H

#include <pvcore/general.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVElementFilter.h>

namespace PVFilter {

class LibFilterDecl PVChunkFilterByElt : public PVChunkFilter {
public:
	PVChunkFilterByElt(PVElementFilter_f elt_filter);
	virtual PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVElementFilter_f _elt_filter;
	mutable PVRow _n_elts_invalid;
};

}

#endif
