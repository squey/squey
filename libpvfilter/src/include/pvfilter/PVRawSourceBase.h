#ifndef PVRAWSOURCEBASE_FILE_H
#define PVRAWSOURCEBASE_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVChunk.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVFilterFunction.h>
#include <QString>

namespace PVRush {
	class PVInput;
}

namespace PVFilter {

class LibExport PVRawSourceBase : public PVFilter::PVFilterFunctionBase<PVCore::PVChunk*,void> {
public:
	typedef boost::shared_ptr<PVRawSourceBase> p_type;
public:
	PVRawSourceBase(PVFilter::PVChunkFilter_f src_filter);
	virtual ~PVRawSourceBase();
public:
//	virtual PVCore::PVChunk* operator()() = 0;
	PVFilter::PVChunkFilter_f source_filter();
	// FIXME: PVInput should be in this class (inter-library depedencies problem...) !
	virtual PVRush::PVInput* get_input() { return NULL; }
	PVCore::chunk_index last_elt_index() { return _last_elt_index; }
	virtual void seek_begin() {};
	virtual QString human_name() { return QString("undefined"); }
protected:
	PVFilter::PVChunkFilter_f _src_filter;
	mutable PVCore::chunk_index _last_elt_index; // Local file index of the last element of that source. Can correspond to a number of lines
};

typedef PVRawSourceBase::p_type PVRawSourceBase_p;

}

#endif
