/**
 * \file PVNrawOutput.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVNRAWOUTPUT_FILE_H
#define PVNRAWOUTPUT_FILE_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVOutput.h>
#include <pvkernel/rush/PVNraw.h>

#include <functional>
#include <vector>

namespace PVRush {

class LibKernelDecl PVNrawOutput : public PVRush::PVOutput {
public:
	typedef std::map<PVRow,chunk_index> map_pvrow;
public:
	PVNrawOutput();

public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVChunk* out);
	map_pvrow const& get_pvrow_index_map() const;
	void clear_pvrow_index_map();

	void set_nraw_dest(PVNraw& nraw) { _nraw_dest = &nraw; }

public:
	PVNraw const& nraw_dest() const { assert(_nraw_dest); return *_nraw_dest; }	
	PVNraw& nraw_dest() { assert(_nraw_dest); return *_nraw_dest; }	

protected:
	void job_has_finished();
	
protected:
	PVNraw* _nraw_dest;
	map_pvrow _pvrow_chunk_idx;
	PVRow _nraw_cur_index;

	CLASS_FILTER_NONREG(PVNrawOutput)
};

}

#endif
