#include <pvkernel/filter/PVRawSourceBase.h>


PVFilter::PVRawSourceBase::PVRawSourceBase(PVFilter::PVChunkFilter_f src_filter) :
		_src_filter(src_filter)
{
	_last_elt_index = 0;
}

PVFilter::PVRawSourceBase::~PVRawSourceBase()
{
}

PVFilter::PVChunkFilter_f PVFilter::PVRawSourceBase::source_filter()
{
	return _src_filter;
};


