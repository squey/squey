#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInput.h>

PVRush::PVRawSourceBase::PVRawSourceBase(PVInput_p input, PVFilter::PVChunkFilter_f src_filter) :
	_src_filter(src_filter),
	_input(input)
{
	_last_elt_index = 0;
}

PVRush::PVRawSourceBase::~PVRawSourceBase()
{
}

PVFilter::PVChunkFilter_f PVRush::PVRawSourceBase::source_filter()
{
	return _src_filter;
};

void PVRush::PVRawSourceBase::seek_begin()
{
	_input->seek_begin();
}

QString PVRush::PVRawSourceBase::human_name()
{
	return _input->human_name();
}
