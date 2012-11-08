/**
 * \file PVOriginalAxisIndexType.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>

PVCore::PVOriginalAxisIndexType::PVOriginalAxisIndexType(bool append_none_axis)
{
	_origin_axis_index = -1;
}

PVCore::PVOriginalAxisIndexType::PVOriginalAxisIndexType(int origin_axis_index, bool append_none_axis)
{
	_origin_axis_index = origin_axis_index;
	_append_none_axis = append_none_axis;
}

int PVCore::PVOriginalAxisIndexType::get_original_index() const
{
	return _origin_axis_index;
}

bool PVCore::PVOriginalAxisIndexType::get_append_none_axis() const
{
	return _append_none_axis;
}
