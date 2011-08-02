//! \file PVAxisIndexType.cpp
//! $Id: PVAxisIndexType.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexType.h>

PVCore::PVAxisIndexType::PVAxisIndexType(bool append_none_axis)
{
	_origin_axis_index = -1;
	_append_none_axis = append_none_axis;
}

PVCore::PVAxisIndexType::PVAxisIndexType(int origin_axis_index, bool append_none_axis)
{
	_origin_axis_index = origin_axis_index;
	_append_none_axis = append_none_axis;
}

int PVCore::PVAxisIndexType::get_original_index()
{
	return _origin_axis_index;
}

bool PVCore::PVAxisIndexType::get_append_none_axis()
{
	return _append_none_axis;
}
