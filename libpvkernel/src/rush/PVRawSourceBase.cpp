/**
 * \file PVRawSourceBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
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
