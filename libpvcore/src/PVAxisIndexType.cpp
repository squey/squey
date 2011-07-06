//! \file PVAxisIndexType.cpp
//! $Id: PVAxisIndexType.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/general.h>
#include <pvcore/PVAxisIndexType.h>

PVCore::PVAxisIndexType::PVAxisIndexType()
{
	_origin_axis_index = -1;
}

PVCore::PVAxisIndexType::PVAxisIndexType(int origin_axis_index)
{
	_origin_axis_index = origin_axis_index;
}

int PVCore::PVAxisIndexType::get_original_index()
{
	return _origin_axis_index;
}
