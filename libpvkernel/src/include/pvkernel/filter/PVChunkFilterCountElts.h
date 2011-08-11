//! \file PVChunkFilterCountElts.h
//! $Id: PVChunkFilterCountElts.h 3187 2011-06-21 11:20:33Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVCHUNKFILTERCOUNTELTS_H
#define PVFILTER_PVCHUNKFILTERCOUNTELTS_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <map>

namespace PVFilter {

class LibKernelDecl PVChunkFilterCountElts : public PVChunkFilter {

public:
	PVChunkFilterCountElts(bool* done_ptr);
	void done_when(chunk_index n);

public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
	inline chunk_index n_elts_done() const { return _n_elts; }

protected:
	mutable chunk_index _n_elts;
	chunk_index _done_when;
	bool* _done_ptr;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterCountElts)
};

}

#endif
