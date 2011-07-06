//! \file PVChunkFilterCountElts.h
//! $Id: PVChunkFilterCountElts.h 3187 2011-06-21 11:20:33Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVCHUNKFILTERCOUNTELTS_H
#define PVFILTER_PVCHUNKFILTERCOUNTELTS_H

#include <pvcore/general.h>
#include <pvcore/types.h>
#include <pvfilter/PVChunkFilter.h>

#include <map>

namespace PVFilter {

class LibExport PVChunkFilterCountElts : public PVChunkFilter {

public:
	PVChunkFilterCountElts(bool* done_ptr);
	void done_when(PVCore::chunk_index n);

public:
	virtual PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);
	inline PVCore::chunk_index n_elts_done() const { return _n_elts; }

protected:
	mutable PVCore::chunk_index _n_elts;
	PVCore::chunk_index _done_when;
	bool* _done_ptr;
};

}

#endif
