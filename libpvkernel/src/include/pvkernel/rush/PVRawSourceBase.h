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
	PVRawSourceBase(PVInput_p input, PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVRawSourceBase();
public:
	PVFilter::PVChunkFilter_f source_filter();
	PVInput_p get_input() { return _input; }
	PVCore::chunk_index last_elt_index() { return _last_elt_index; }
	virtual QString human_name();
	virtual void seek_begin();

protected:
	PVFilter::PVChunkFilter_f _src_filter;
	mutable PVCore::chunk_index _last_elt_index; // Local file index of the last element of that source. Can correspond to a number of lines
	PVInput_p _input;
};

}

#endif
