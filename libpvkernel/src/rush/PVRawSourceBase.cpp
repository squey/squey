/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVInput.h>

PVRush::PVRawSourceBase::PVRawSourceBase(PVFilter::PVChunkFilter_f src_filter) :
	_src_filter(src_filter),
	_ncols_to_reserve(1)
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
