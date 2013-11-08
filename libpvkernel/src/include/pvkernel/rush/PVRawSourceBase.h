/**
 * \file PVRawSourceBase.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRAWSOURCEBASE_FILE_H
#define PVRAWSOURCEBASE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/rush/PVInput_types.h>
#include <QString>
#include <boost/shared_ptr.hpp>

#include <pvkernel/rush/PVRawSourceBase_types.h>

namespace PVRush {

class LibKernelDecl PVRawSourceBase : public PVFilter::PVFilterFunctionBase<PVCore::PVChunk*,void> {
public:
	typedef PVRawSourceBase_p p_type;
public:
	PVRawSourceBase(PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVRawSourceBase();
private:
	PVRawSourceBase(const PVRawSourceBase& src) :
		PVFilter::PVFilterFunctionBase<PVCore::PVChunk*,void>(src)
	{
		assert(false);
	}

public:
	virtual void release_input() {}

public:
	PVFilter::PVChunkFilter_f source_filter();
	chunk_index last_elt_index() { return _last_elt_index; }
	void set_number_cols_to_reserve(PVCol col)
	{
		if (col == 0) {
			col = 1;
		}
		_ncols_to_reserve = col;
	}
	PVCol get_number_cols_to_reserve() const { return _ncols_to_reserve; }
	virtual QString human_name() = 0;
	virtual void seek_begin() = 0;
	virtual bool seek(input_offset off) = 0;
	virtual void prepare_for_nelts(chunk_index nelts) = 0;
	virtual PVCore::PVChunk* operator()() = 0;
	virtual input_offset get_input_offset_from_index(chunk_index idx, chunk_index& known_idx) = 0;

protected:
	PVFilter::PVChunkFilter_f _src_filter;
	mutable chunk_index _last_elt_index; // Local file index of the last element of that source. Can correspond to a number of lines
	PVCol _ncols_to_reserve;
};

}

#endif
