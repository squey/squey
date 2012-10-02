/**
 * \file PVAxisIndexType.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxisIndexType.h>

PVCore::PVAxisIndexType::PVAxisIndexType(bool append_none_axis)
{
	_origin_axis_index = -1;
	_append_none_axis = append_none_axis;
}

PVCore::PVAxisIndexType::PVAxisIndexType(int origin_axis_index, bool append_none_axis, int axis_index)
{
	_origin_axis_index = origin_axis_index;
	_append_none_axis = append_none_axis;
	_axis_index = axis_index;
}

int PVCore::PVAxisIndexType::get_original_index()
{
	return _origin_axis_index;
}

bool PVCore::PVAxisIndexType::get_append_none_axis()
{
	return _append_none_axis;
}

int PVCore::PVAxisIndexType::get_axis_index()
{
	return _axis_index;
}
