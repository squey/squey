/**
 * \file PVChunkFilterCountElts.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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
	inline chunk_index n_elts_invalid() const { return _n_elts_invalid; }

protected:
	mutable chunk_index _n_elts;
	mutable chunk_index _n_elts_invalid;
	chunk_index _done_when;
	bool* _done_ptr;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterCountElts)
};

}

#endif
