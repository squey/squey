//! \file PVChunkFilterByEltRestoreInvalid.h
//! $Id: PVChunkFilterByEltRestoreInvalid.h 3129 2011-06-14 09:47:24Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVCHUNKFILTERBYELTRESTOREINVALID_H
#define PVFILTER_PVCHUNKFILTERBYELTRESTOREINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

class LibKernelDecl PVChunkFilterByEltRestoreInvalid: public PVChunkFilter {
public:
	PVChunkFilterByEltRestoreInvalid(PVElementFilter_f elt_filter);
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
protected:
	mutable PVElementFilter_f _elt_filter;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterByEltRestoreInvalid)
};

}

#endif
